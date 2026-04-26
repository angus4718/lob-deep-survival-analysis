from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from .common import OutputPaths, dump_json
    from .dynamic_artifact_adapter import LoadedDynamicArtifact
    from .dynamic_backtest import DynamicBacktestResult
    from .dynamic_data_adapter import DynamicDatasetBundle
except ImportError:
    from common import OutputPaths, dump_json
    from dynamic_artifact_adapter import LoadedDynamicArtifact
    from dynamic_backtest import DynamicBacktestResult
    from dynamic_data_adapter import DynamicDatasetBundle


def write_dynamic_backtest_outputs(
    output_paths: OutputPaths,
    artifact: LoadedDynamicArtifact,
    bundle: DynamicDatasetBundle,
    result: DynamicBacktestResult,
    cli_config: dict[str, Any],
) -> None:
    result.summary.to_csv(output_paths.summary_path, index=False)
    result.daily_summary.to_csv(output_paths.daily_summary_path, index=False)
    result.trades.to_csv(output_paths.trades_path, index=False)

    dump_json(
        output_paths.config_path,
        {
            "cli_config": cli_config,
            "artifact": artifact.to_dict(),
            "dataset": {
                "dataset_prefix": str(bundle.dataset_prefix),
                "split_name": bundle.split_name,
                "feature_source": bundle.feature_source,
                "schema_summary": bundle.schema_summary,
                "available_post_trade_windows_ms": bundle.available_post_trade_windows_ms,
                "training_priors": bundle.training_priors,
                "num_eval_rows": int(len(bundle.eval_frame)),
                "paths": {
                    "split_parquet": str(bundle.paths.split_parquet_path),
                    "preprocessed_npz": str(bundle.paths.preprocessed_npz_path),
                    "sample_manifest": str(bundle.paths.sample_manifest_path),
                    "manifest_parquet": str(bundle.paths.manifest_parquet_path),
                    "order_store": str(bundle.paths.order_store_path),
                },
            },
            "backtest": result.config,
        },
    )

    figure_paths = _write_figures(output_paths, result)
    output_paths.report_path.write_text(
        _build_report_markdown(
            artifact=artifact,
            bundle=bundle,
            result=result,
            cli_config=cli_config,
            figure_paths=figure_paths,
        )
    )


def _write_figures(output_paths: OutputPaths, result: DynamicBacktestResult) -> list[Path]:
    if plt is None or result.summary.empty:
        return []

    figure_paths: list[Path] = []
    summary = result.summary.sort_values("strategy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(summary["strategy"], summary["mean_implementation_shortfall"], color="#d62728")
    axes[0].set_title("Mean Implementation Shortfall")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(summary["strategy"], summary["mean_realized_pnl"], color="#2ca02c")
    axes[1].set_title("Mean Realized PnL")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    summary_fig = output_paths.figures_dir / "summary_metrics.png"
    fig.savefig(summary_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(summary_fig)

    if not result.daily_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for strategy, group in result.daily_summary.groupby("strategy", sort=False):
            ordered = group.sort_values("entry_date")
            ax.plot(ordered["entry_date"], ordered["mean_realized_pnl"], marker="o", label=strategy)
        ax.set_title("Daily Mean Realized PnL")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        daily_fig = output_paths.figures_dir / "daily_realized_pnl.png"
        fig.savefig(daily_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(daily_fig)

    return figure_paths


def _build_report_markdown(
    artifact: LoadedDynamicArtifact,
    bundle: DynamicDatasetBundle,
    result: DynamicBacktestResult,
    cli_config: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    lines: list[str] = []
    lines.append(f"# Dynamic Backtest Report: {result.artifact_name}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Dataset prefix: `{bundle.dataset_prefix}`")
    lines.append(f"- Eval split: `{bundle.split_name}`")
    lines.append(f"- Eval rows: `{len(bundle.eval_frame):,}`")
    lines.append(f"- Model: `{artifact.metadata.model_name}`")
    lines.append(f"- Framework: `{artifact.metadata.framework}`")
    lines.append(f"- Horizon: `{result.config['horizon_s']}` seconds")
    lines.append(f"- Re-evaluation interval: `{result.config['reeval_interval_s']}`")
    lines.append(f"- Delta proc: `{result.config['delta_proc_s']}`")
    lines.append(f"- Delta lat: `{result.config['delta_lat_s']}`")
    lines.append("")
    lines.append("## Strategy Summary")
    lines.append("")
    lines.extend(_markdown_table(result.summary))
    lines.append("")
    lines.append("## Proposal Alignment Notes")
    lines.append("")
    lines.append("- Utility follows the proposal form: positive fill probability minus risk-weighted toxic probability.")
    lines.append("- Dynamic policy re-evaluates resting orders in walk-forward order time.")
    lines.append("- If utility drops below threshold while the order is resting, the policy cancels and crosses the spread.")
    lines.append("- Latency is modeled with `delta_proc_s` and `delta_lat_s` before the market-taking action becomes effective.")
    lines.append("- FIFO queue realism is inherited from the precomputed virtual-order tracking in the dynamic dataset.")
    lines.append("")
    if not result.daily_summary.empty:
        lines.append("## Daily Summary")
        lines.append("")
        lines.extend(_markdown_table(result.daily_summary.head(20)))
        if len(result.daily_summary) > 20:
            lines.append("")
            lines.append(f"_Showing first 20 of {len(result.daily_summary)} daily rows._")
        lines.append("")
    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for figure_path in figure_paths:
            lines.append(f"- [{figure_path.name}]({figure_path})")
        lines.append("")
    lines.append("## Saved Outputs")
    lines.append("")
    lines.append(f"- Summary CSV: `{cli_config['output_dir']}/summary.csv`")
    lines.append(f"- Daily summary CSV: `{cli_config['output_dir']}/daily_summary.csv`")
    lines.append(f"- Trade-level CSV: `{cli_config['output_dir']}/trades.csv`")
    lines.append(f"- Config JSON: `{cli_config['output_dir']}/config.json`")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(_fmt_float)
    headers = [str(col) for col in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return lines


def _fmt_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"
