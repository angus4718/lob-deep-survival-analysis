#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .common import build_output_paths
    from .dynamic_artifact_adapter import load_dynamic_artifact, load_dynamic_artifact_metadata
    from .dynamic_backtest import DynamicBacktestConfig, run_dynamic_backtest
    from .dynamic_data_adapter import load_dynamic_dataset
    from .dynamic_report_generator import write_dynamic_backtest_outputs
except ImportError:
    from common import build_output_paths
    from dynamic_artifact_adapter import load_dynamic_artifact, load_dynamic_artifact_metadata
    from dynamic_backtest import DynamicBacktestConfig, run_dynamic_backtest
    from dynamic_data_adapter import load_dynamic_dataset
    from dynamic_report_generator import write_dynamic_backtest_outputs


DEFAULT_ARTIFACT_BASE_NET = Path(
    "/ocean/projects/cis260122p/shared/artifacts/dynamic/dynamic_deephit_gru_transformer_base_net.pt"
)
DEFAULT_ARTIFACT_META = Path(
    "/ocean/projects/cis260122p/shared/artifacts/dynamic/dynamic_deephit_gru_transformer_meta.pt"
)
DEFAULT_DATASET_PREFIX = Path(
    "/ocean/projects/cis260122p/shared/data/datasets/labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101"
)
DEFAULT_PROJECT_ROOT = Path(
    "/ocean/projects/cis260122p/shared/lob-deep-survival-analysis-main"
)
DEFAULT_OUTPUT_DIR = Path(
    "/ocean/projects/cis260122p/hwang71/backtest/outputs/output_dynamic_deephit_gru_transformer"
)
DEFAULT_HORIZON_S = 60.0
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_DECISION_THRESHOLD = 0.0
DEFAULT_REEVAL_INTERVAL_S = None
DEFAULT_DELTA_PROC_S = 0.0
DEFAULT_DELTA_LAT_S = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a proposal-aligned dynamic backtest for Dynamic-DeepHit artifacts."
    )
    parser.add_argument("--artifact-base-net", type=Path, default=DEFAULT_ARTIFACT_BASE_NET)
    parser.add_argument("--artifact-meta", type=Path, default=DEFAULT_ARTIFACT_META)
    parser.add_argument(
        "--dataset-prefix",
        type=Path,
        default=DEFAULT_DATASET_PREFIX,
        help="Base path prefix before _train/_val/_test and _dynamic_* suffixes.",
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    parser.add_argument("--risk-aversion", type=float, default=DEFAULT_RISK_AVERSION)
    parser.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    parser.add_argument("--reeval-interval-s", type=float, default=DEFAULT_REEVAL_INTERVAL_S)
    parser.add_argument("--delta-proc-s", type=float, default=DEFAULT_DELTA_PROC_S)
    parser.add_argument("--delta-lat-s", type=float, default=DEFAULT_DELTA_LAT_S)
    parser.add_argument(
        "--strategies",
        type=str,
        default="model,always_market,always_limit,training_prior",
    )
    parser.add_argument("--markout-window-ms", type=int, default=None)
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    print(f"Loading dynamic artifact metadata from: {args.artifact_meta}", flush=True)
    metadata = load_dynamic_artifact_metadata(args.artifact_meta)
    print(
        "[dynamic_runner] artifact metadata: "
        f"model_name={metadata.model_name} framework={metadata.framework} "
        f"lookback={metadata.lookback_steps} output_steps={metadata.output_steps} "
        f"events={metadata.event_names} time_grid_max={float(metadata.time_grid[-1]):.6f}",
        flush=True,
    )

    print(
        f"Preparing dynamic dataset split={args.split}, lookback={metadata.lookback_steps}, horizon={args.horizon_s}s",
        flush=True,
    )
    bundle = load_dynamic_dataset(
        dataset_prefix=args.dataset_prefix,
        lookback_steps=metadata.lookback_steps,
        split_name=args.split,
        time_grid=metadata.time_grid,
        project_root=args.project_root,
        verbose=True,
    )
    print(
        "[dynamic_runner] dataset ready: "
        f"feature_source={bundle.feature_source} feature_dim={bundle.feature_dim} "
        f"eval_rows={len(bundle.eval_frame)} split_path={bundle.paths.split_parquet_path}",
        flush=True,
    )

    print(f"Loading dynamic artifact: {args.artifact_base_net}", flush=True)
    artifact = load_dynamic_artifact(
        artifact_base_net_path=args.artifact_base_net,
        artifact_meta_path=args.artifact_meta,
        device=args.device,
    )
    print(
        f"[dynamic_runner] artifact loaded on device={artifact.device} "
        f"init_kwargs={artifact.init_kwargs}",
        flush=True,
    )

    print("Running dynamic model inference...", flush=True)
    cif = artifact.predict_cif(bundle.x_eval, batch_size=args.batch_size)
    print(
        f"[dynamic_runner] cif_shape={tuple(int(v) for v in cif.shape)}",
        flush=True,
    )

    config = DynamicBacktestConfig(
        horizon_s=args.horizon_s,
        risk_aversion=args.risk_aversion,
        decision_threshold=args.decision_threshold,
        split_name=args.split,
        reeval_interval_s=args.reeval_interval_s,
        delta_proc_s=args.delta_proc_s,
        delta_lat_s=args.delta_lat_s,
        strategies=strategies,
        markout_window_ms=args.markout_window_ms,
        market_future_mid_mode=args.market_future_mid_mode,
        cleanup_future_mid_mode=args.cleanup_future_mid_mode,
    )

    print("Running proposal-aligned dynamic backtest...", flush=True)
    result = run_dynamic_backtest(
        bundle=bundle,
        artifact=artifact,
        cif=cif,
        config=config,
    )

    output_paths = build_output_paths(args.output_dir)
    cli_config = {
        "artifact_base_net": str(args.artifact_base_net.resolve()),
        "artifact_meta": str(args.artifact_meta.resolve()),
        "dataset_prefix": str(args.dataset_prefix.resolve()),
        "project_root": str(args.project_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split": args.split,
        "horizon_s": args.horizon_s,
        "risk_aversion": args.risk_aversion,
        "decision_threshold": args.decision_threshold,
        "reeval_interval_s": args.reeval_interval_s,
        "delta_proc_s": args.delta_proc_s,
        "delta_lat_s": args.delta_lat_s,
        "strategies": strategies,
        "markout_window_ms": args.markout_window_ms,
        "market_future_mid_mode": args.market_future_mid_mode,
        "cleanup_future_mid_mode": args.cleanup_future_mid_mode,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
    }

    write_dynamic_backtest_outputs(
        output_paths=output_paths,
        artifact=artifact,
        bundle=bundle,
        result=result,
        cli_config=cli_config,
    )

    print(f"Dynamic backtest completed. Outputs written to: {output_paths.root}")
    print(f"Summary CSV: {output_paths.summary_path}")
    print(f"Report MD : {output_paths.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
