#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .artifact_adapter import load_artifact, load_artifact_metadata
    from .common import build_output_paths
    from .data_adapter import load_standardized_static_dataset
    from .report_generator import write_backtest_outputs
    from .static_backtest import StaticBacktestConfig, run_static_backtest
except ImportError:
    from artifact_adapter import load_artifact, load_artifact_metadata
    from common import build_output_paths
    from data_adapter import load_standardized_static_dataset
    from report_generator import write_backtest_outputs
    from static_backtest import StaticBacktestConfig, run_static_backtest


DEFAULT_ARTIFACT_BASE_NET = Path(
    "/ocean/projects/cis260122p/shared/artifacts/baseline/standardized_deephit_transformer_base_net.pt"
)
DEFAULT_ARTIFACT_META = Path(
    "/ocean/projects/cis260122p/shared/artifacts/baseline/standardized_deephit_transformer_meta.pt"
)
DEFAULT_DATASET = Path(
    "/ocean/projects/cis260122p/shared/data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/ocean/projects/cis260122p/hwang71/backtest/outputs/output_standardized_deephit_transformer"
)
DEFAULT_PROJECT_ROOT = Path("/ocean/projects/cis260122p/hwang71/lob-deep-survival-analysis-main")
DEFAULT_MAX_TIME_S = 60.0
DEFAULT_HORIZON_S = 60.0
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_DECISION_THRESHOLD = 0.0


def parse_args() -> argparse.Namespace:
    final_project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a generic artifact-driven static backtest for standardized DeepHit models."
    )
    parser.add_argument(
        "--artifact-base-net",
        type=Path,
        default=DEFAULT_ARTIFACT_BASE_NET,
        help="Path to the artifact base_net .pt file.",
    )
    parser.add_argument(
        "--artifact-meta",
        type=Path,
        default=DEFAULT_ARTIFACT_META,
        help="Path to the artifact metadata .pt file.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the labeled parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write backtest outputs.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repo root containing src/models.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Dataset split to evaluate after reconstructing the training preprocessing.",
    )
    parser.add_argument(
        "--max-time-s",
        type=float,
        default=DEFAULT_MAX_TIME_S,
        help="Recensor late events the same way as the standardized static notebook.",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=DEFAULT_HORIZON_S,
        help="Decision horizon in seconds. Defaults to 60.0 for cross-model comparability.",
    )
    parser.add_argument("--risk-aversion", type=float, default=DEFAULT_RISK_AVERSION)
    parser.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    parser.add_argument(
        "--strategies",
        type=str,
        default="model,always_market,always_limit,training_prior",
        help="Comma-separated strategy list.",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference, e.g. cpu or cuda.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    print(f"Loading artifact metadata from: {args.artifact_meta}", flush=True)
    metadata = load_artifact_metadata(args.artifact_meta)
    horizon_s = float(args.horizon_s)

    print(
        f"Preparing dataset split={args.split}, horizon={horizon_s}s, device={args.device}",
        flush=True,
    )
    dataset_bundle = load_standardized_static_dataset(
        dataset_path=args.dataset,
        lookback_steps=metadata.lookback_steps,
        max_time_s=args.max_time_s,
        split_name=args.split,
    )

    print(f"Loading model artifact: {args.artifact_base_net}", flush=True)
    artifact = load_artifact(
        artifact_base_net_path=args.artifact_base_net,
        artifact_meta_path=args.artifact_meta,
        project_root=args.project_root,
        feature_dim=dataset_bundle.feature_dim,
        device=args.device,
    )

    print("Running model inference...", flush=True)
    cif = artifact.predict_cif(dataset_bundle.x_eval, batch_size=args.batch_size)

    backtest_config = StaticBacktestConfig(
        horizon_s=horizon_s,
        risk_aversion=args.risk_aversion,
        decision_threshold=args.decision_threshold,
        split_name=args.split,
        markout_window_ms=args.markout_window_ms,
        strategies=strategies,
        cleanup_price_mode=args.cleanup_price_mode,
        market_future_mid_mode=args.market_future_mid_mode,
        cleanup_future_mid_mode=args.cleanup_future_mid_mode,
        use_execution_quote_for_cleanup=True,
    )

    print("Running static backtest...", flush=True)
    result = run_static_backtest(
        bundle=dataset_bundle,
        artifact=artifact,
        cif=cif,
        config=backtest_config,
    )

    output_paths = build_output_paths(args.output_dir)
    cli_config = {
        "artifact_base_net": str(args.artifact_base_net.resolve()),
        "artifact_meta": str(args.artifact_meta.resolve()),
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "project_root": str(args.project_root.resolve()),
        "split": args.split,
        "max_time_s": args.max_time_s,
        "horizon_s": horizon_s,
        "risk_aversion": args.risk_aversion,
        "decision_threshold": args.decision_threshold,
        "strategies": strategies,
        "markout_window_ms": args.markout_window_ms,
        "cleanup_price_mode": args.cleanup_price_mode,
        "market_future_mid_mode": args.market_future_mid_mode,
        "cleanup_future_mid_mode": args.cleanup_future_mid_mode,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
    }

    write_backtest_outputs(
        output_paths=output_paths,
        artifact=artifact,
        bundle=dataset_bundle,
        result=result,
        cli_config=cli_config,
    )

    print(f"Backtest completed. Outputs written to: {output_paths.root}")
    print(f"Summary CSV: {output_paths.summary_path}")
    print(f"Report MD : {output_paths.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
