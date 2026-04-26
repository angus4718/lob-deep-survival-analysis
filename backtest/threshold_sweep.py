#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .artifact_adapter import load_artifact, load_artifact_metadata
    from .common import dump_json
    from .data_adapter import DatasetBundle, load_standardized_static_dataset
    from .static_backtest import StaticBacktestConfig, run_static_backtest
except ImportError:
    from artifact_adapter import load_artifact, load_artifact_metadata
    from common import dump_json
    from data_adapter import DatasetBundle, load_standardized_static_dataset
    from static_backtest import StaticBacktestConfig, run_static_backtest


DEFAULT_ARTIFACT_DIR = Path("/ocean/projects/cis260122p/shared/artifacts/baseline")
DEFAULT_DATASET = Path(
    "/ocean/projects/cis260122p/shared/data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101.parquet"
)
DEFAULT_PROJECT_ROOT = Path("/ocean/projects/cis260122p/hwang71/lob-deep-survival-analysis-main")
DEFAULT_OUTPUT_DIR = Path("/ocean/projects/cis260122p/hwang71/backtest/threshold_sweeps")
DEFAULT_ARTIFACT_PATTERN = "standardized_deephit_*_base_net.pt"
DEFAULT_EXCLUDED_MODEL_NAMES = "mamba" #need gpu, will run separately
DEFAULT_THRESHOLDS = "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6"
DEFAULT_SELECTION_SPLIT = "val"
DEFAULT_HORIZON_S = 60.0
DEFAULT_MAX_TIME_S = 60.0
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_CANCEL_RATE_CAP = 0.20
DEFAULT_STRATEGIES = "model,always_market,always_limit,training_prior"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 1024


def parse_args() -> argparse.Namespace:
    final_project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep decision thresholds for standardized static artifacts on the validation split, "
            "and select the best threshold per artifact."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing standardized static *_base_net.pt and *_meta.pt artifacts.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset used for the threshold sweep.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repo root containing src/models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write sweep summaries and selected thresholds.",
    )
    parser.add_argument(
        "--artifact-pattern",
        type=str,
        default=DEFAULT_ARTIFACT_PATTERN,
        help="Glob pattern used inside artifact-dir to discover artifacts.",
    )
    parser.add_argument(
        "--exclude-model-names",
        type=str,
        default=DEFAULT_EXCLUDED_MODEL_NAMES,
        help="Comma-separated model_name values to skip. Defaults to excluding mamba.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated threshold grid.",
    )
    parser.add_argument(
        "--selection-split",
        choices=["val", "test"],
        default=DEFAULT_SELECTION_SPLIT,
        help="Split used to select the best threshold. Recommended: val.",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=DEFAULT_HORIZON_S,
        help="Fixed decision horizon in seconds for every artifact.",
    )
    parser.add_argument(
        "--max-time-s",
        type=float,
        default=DEFAULT_MAX_TIME_S,
        help="Re-censor late uncensored events when reconstructing the dataset.",
    )
    parser.add_argument("--risk-aversion", type=float, default=DEFAULT_RISK_AVERSION)
    parser.add_argument(
        "--cancel-rate-cap",
        type=float,
        default=DEFAULT_CANCEL_RATE_CAP,
        help="Prefer thresholds whose model cancel_rate is at most this value on the selection split.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=DEFAULT_STRATEGIES,
        help="Comma-separated strategy list passed into the backtest engine.",
    )
    parser.add_argument(
        "--markout-window-ms",
        type=int,
        default=None,
        help="Post-trade window to use for passive-fill RPnL. Defaults to the smallest available window.",
    )
    parser.add_argument(
        "--cleanup-price-mode",
        choices=["entry_quote", "execution_quote"],
        default="entry_quote",
    )
    parser.add_argument(
        "--market-future-mid-mode",
        choices=["entry_mid", "execution_mid"],
        default="entry_mid",
    )
    parser.add_argument(
        "--cleanup-future-mid-mode",
        choices=["entry_mid", "execution_mid"],
        default="entry_mid",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    excluded_model_names = {
        item.strip() for item in args.exclude_model_names.split(",") if item.strip()
    }

    artifact_pairs = discover_artifact_pairs(
        args.artifact_dir,
        args.artifact_pattern,
        excluded_model_names=excluded_model_names,
    )
    if not artifact_pairs:
        raise FileNotFoundError(
            f"No artifacts found in {args.artifact_dir} matching {args.artifact_pattern!r}."
        )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selection_cache: dict[tuple[int, str], DatasetBundle] = {}

    sweep_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for base_net_path, meta_path in artifact_pairs:
        metadata = load_artifact_metadata(meta_path)
        artifact_name = base_net_path.stem.removesuffix("_base_net")

        selection_bundle = get_dataset_bundle(
            cache=selection_cache,
            dataset_path=args.dataset,
            lookback_steps=metadata.lookback_steps,
            max_time_s=args.max_time_s,
            split_name=args.selection_split,
        )

        artifact = load_artifact(
            artifact_base_net_path=base_net_path,
            artifact_meta_path=meta_path,
            project_root=args.project_root,
            feature_dim=selection_bundle.feature_dim,
            device=args.device,
        )
        selection_cif = artifact.predict_cif(selection_bundle.x_eval, batch_size=args.batch_size)

        model_sweep_rows = []
        for threshold in thresholds:
            config = build_backtest_config(
                horizon_s=args.horizon_s,
                risk_aversion=args.risk_aversion,
                decision_threshold=threshold,
                split_name=args.selection_split,
                markout_window_ms=args.markout_window_ms,
                strategies=strategies,
                cleanup_price_mode=args.cleanup_price_mode,
                market_future_mid_mode=args.market_future_mid_mode,
                cleanup_future_mid_mode=args.cleanup_future_mid_mode,
            )
            result = run_static_backtest(
                bundle=selection_bundle,
                artifact=artifact,
                cif=selection_cif,
                config=config,
                artifact_name=artifact_name,
            )
            summary = result.summary.copy()
            summary["threshold"] = threshold
            summary["selection_split"] = args.selection_split
            summary["artifact_base_net"] = str(base_net_path.resolve())
            summary["artifact_meta"] = str(meta_path.resolve())
            sweep_rows.extend(summary.to_dict(orient="records"))

            model_row = summary.loc[summary["strategy"] == "model"].iloc[0].to_dict()
            model_sweep_rows.append(model_row)

        best_row = choose_best_threshold(
            model_rows=pd.DataFrame(model_sweep_rows),
            cancel_rate_cap=args.cancel_rate_cap,
        )
        selected_threshold = float(best_row["threshold"])

        selected_payload = {
            "artifact_name": artifact_name,
            "artifact_base_net": str(base_net_path.resolve()),
            "artifact_meta": str(meta_path.resolve()),
            "selected_threshold": selected_threshold,
            "selection_split": args.selection_split,
            "cancel_rate_cap": args.cancel_rate_cap,
            "selection_mean_realized_pnl": float(best_row["mean_realized_pnl"]),
            "selection_mean_implementation_shortfall": float(
                best_row["mean_implementation_shortfall"]
            ),
            "selection_cancel_rate": float(best_row["cancel_rate"]),
            "selection_limit_decision_rate": float(best_row["limit_decision_rate"]),
            "selection_market_entry_rate": float(best_row["market_entry_rate"]),
            "selection_passive_fill_rate": float(best_row["passive_fill_rate"]),
            "selection_toxic_fill_rate_overall": float(best_row["toxic_fill_rate_overall"]),
            "selection_avg_utility": float(best_row["avg_utility"]),
        }
        selected_rows.append(selected_payload)

        print(
            f"[{artifact_name}] selected threshold {selected_threshold:.4f} on {args.selection_split}"
        )

    sweep_frame = pd.DataFrame(sweep_rows)
    if not sweep_frame.empty:
        sweep_frame = sweep_frame.sort_values(["artifact_name", "threshold", "strategy"]).reset_index(drop=True)
    selected_frame = pd.DataFrame(selected_rows).sort_values("artifact_name").reset_index(drop=True)

    sweep_path = output_root / "selection_threshold_sweep.csv"
    selected_path = output_root / "best_thresholds.csv"
    selected_json_path = output_root / "best_thresholds.json"
    sweep_frame.to_csv(sweep_path, index=False)
    selected_frame.to_csv(selected_path, index=False)
    dump_json(
        selected_json_path,
        {
            row["artifact_name"]: {
                "selected_threshold": row["selected_threshold"],
                "artifact_base_net": row["artifact_base_net"],
                "artifact_meta": row["artifact_meta"],
            }
            for row in selected_rows
        },
    )

    dump_json(
        output_root / "sweep_config.json",
        {
            "artifact_dir": str(args.artifact_dir.resolve()),
            "dataset": str(args.dataset.resolve()),
            "project_root": str(args.project_root.resolve()),
            "output_dir": str(output_root),
            "artifact_pattern": args.artifact_pattern,
            "exclude_model_names": sorted(excluded_model_names),
            "thresholds": thresholds,
            "selection_split": args.selection_split,
            "horizon_s": args.horizon_s,
            "max_time_s": args.max_time_s,
            "risk_aversion": args.risk_aversion,
            "cancel_rate_cap": args.cancel_rate_cap,
            "strategies": strategies,
            "markout_window_ms": args.markout_window_ms,
            "cleanup_price_mode": args.cleanup_price_mode,
            "market_future_mid_mode": args.market_future_mid_mode,
            "cleanup_future_mid_mode": args.cleanup_future_mid_mode,
            "device": args.device,
            "batch_size": args.batch_size,
        },
    )

    print(f"Sweep summary saved to: {sweep_path}")
    print(f"Best thresholds saved to: {selected_path}")
    print(f"Best thresholds JSON saved to: {selected_json_path}")
    return 0


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("Threshold grid is empty.")
    return thresholds


def discover_artifact_pairs(
    artifact_dir: Path,
    pattern: str,
    excluded_model_names: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    artifact_dir = artifact_dir.resolve()
    excluded_model_names = excluded_model_names or set()
    pairs: list[tuple[Path, Path]] = []
    for base_net_path in sorted(artifact_dir.glob(pattern)):
        if not base_net_path.name.endswith("_base_net.pt"):
            continue
        meta_path = base_net_path.with_name(base_net_path.name.replace("_base_net.pt", "_meta.pt"))
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file for {base_net_path.name}: expected {meta_path.name}")
        metadata = load_artifact_metadata(meta_path.resolve())
        if metadata.model_name in excluded_model_names:
            continue
        pairs.append((base_net_path.resolve(), meta_path.resolve()))
    return pairs


def get_dataset_bundle(
    cache: dict[tuple[int, str], DatasetBundle],
    dataset_path: Path,
    lookback_steps: int,
    max_time_s: float,
    split_name: str,
) -> DatasetBundle:
    key = (int(lookback_steps), split_name)
    if key not in cache:
        cache[key] = load_standardized_static_dataset(
            dataset_path=dataset_path,
            lookback_steps=lookback_steps,
            max_time_s=max_time_s,
            split_name=split_name,
        )
    return cache[key]


def build_backtest_config(
    horizon_s: float,
    risk_aversion: float,
    decision_threshold: float,
    split_name: str,
    markout_window_ms: int | None,
    strategies: list[str],
    cleanup_price_mode: str,
    market_future_mid_mode: str,
    cleanup_future_mid_mode: str,
) -> StaticBacktestConfig:
    return StaticBacktestConfig(
        horizon_s=horizon_s,
        risk_aversion=risk_aversion,
        decision_threshold=decision_threshold,
        split_name=split_name,
        markout_window_ms=markout_window_ms,
        strategies=strategies,
        cleanup_price_mode=cleanup_price_mode,
        market_future_mid_mode=market_future_mid_mode,
        cleanup_future_mid_mode=cleanup_future_mid_mode,
        use_execution_quote_for_cleanup=True,
    )


def choose_best_threshold(model_rows: pd.DataFrame, cancel_rate_cap: float) -> dict[str, Any]:
    if model_rows.empty:
        raise ValueError("No model rows available for threshold selection.")

    eligible = model_rows.loc[model_rows["cancel_rate"] <= cancel_rate_cap].copy()
    candidate_pool = eligible if not eligible.empty else model_rows.copy()
    ordered = candidate_pool.sort_values(
        by=[
            "mean_realized_pnl",
            "mean_implementation_shortfall",
            "toxic_fill_rate_overall",
            "threshold",
        ],
        ascending=[False, True, True, True],
    )
    return ordered.iloc[0].to_dict()


if __name__ == "__main__":
    raise SystemExit(main())
