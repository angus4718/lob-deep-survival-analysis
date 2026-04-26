from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from .artifact_adapter import LoadedArtifact
    from .common import OutputPaths, dump_json, format_pct
    from .data_adapter import DatasetBundle
    from .static_backtest import BacktestResult
except ImportError:
    from artifact_adapter import LoadedArtifact
    from common import OutputPaths, dump_json, format_pct
    from data_adapter import DatasetBundle
    from static_backtest import BacktestResult


def write_backtest_outputs(
    output_paths: OutputPaths,
    artifact: LoadedArtifact,
    bundle: DatasetBundle,
    result: BacktestResult,
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
                "dataset_path": str(bundle.dataset_path),
                "split_name": bundle.split_name,
                "available_post_trade_windows_ms": bundle.available_post_trade_windows_ms,
                "training_priors": bundle.training_priors,
                "num_eval_rows": int(len(bundle.eval_frame)),
            },
            "backtest": result.config,
        },
    )

    figure_paths = write_figures(output_paths, result)
    output_paths.report_path.write_text(
        build_report_markdown(
            artifact=artifact,
            bundle=bundle,
            result=result,
            cli_config=cli_config,
            figure_paths=figure_paths,
        )
    )


def write_figures(output_paths: OutputPaths, result: BacktestResult) -> list[Path]:
    if plt is None or result.summary.empty:
        return []

    figure_paths: list[Path] = []

    summary = result.summary.sort_values("strategy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(summary["strategy"], summary["mean_implementation_shortfall"], color="#d62728")
    axes[0].set_title("Mean Implementation Shortfall")
    axes[0].set_ylabel("Signed cost")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(summary["strategy"], summary["mean_realized_pnl"], color="#2ca02c")
    axes[1].set_title("Mean Realized PnL")
    axes[1].set_ylabel("PnL")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    summary_fig = output_paths.figures_dir / "summary_metrics.png"
    fig.savefig(summary_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(summary_fig)

    if not result.daily_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        daily = result.daily_summary.copy()
        for strategy, group in daily.groupby("strategy", sort=False):
            ordered = group.sort_values("entry_date")
            ax.plot(
                ordered["entry_date"],
                ordered["mean_realized_pnl"],
                marker="o",
                linewidth=1.5,
                label=strategy,
            )
        ax.set_title("Daily Mean Realized PnL")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        daily_fig = output_paths.figures_dir / "daily_realized_pnl.png"
        fig.savefig(daily_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(daily_fig)

    return figure_paths


def build_report_markdown(
    artifact: LoadedArtifact,
    bundle: DatasetBundle,
    result: BacktestResult,
    cli_config: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    lines: list[str] = []
    lines.append(f"# Backtest Report: {result.artifact_name}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Dataset: `{bundle.dataset_path}`")
    lines.append(f"- Eval split: `{bundle.split_name}`")
    lines.append(f"- Eval rows: `{len(bundle.eval_frame):,}`")
    lines.append(f"- Model: `{artifact.metadata.model_name}`")
    lines.append(f"- Event names: `{artifact.metadata.event_names}`")
    lines.append(f"- Horizon: `{result.config['horizon_s']}` seconds")
    lines.append(f"- Risk aversion: `{result.config['risk_aversion']}`")
    lines.append(f"- Decision threshold: `{result.config['decision_threshold']}`")
    lines.append("")

    lines.append("## Artifact Interface")
    lines.append("")
    lines.append(f"- Base net: `{artifact.base_net_path}`")
    lines.append(f"- Meta: `{artifact.meta_path}`")
    lines.append(f"- Lookback steps: `{artifact.metadata.lookback_steps}`")
    lines.append(f"- Output steps: `{artifact.metadata.output_steps}`")
    lines.append("")

    lines.append("## Strategy Summary")
    lines.append("")
    lines.extend(_markdown_table(result.summary))
    lines.append("")

    lines.append("## Dataset Notes")
    lines.append("")
    lines.append(f"- Training prior P(FAVORABLE): `{bundle.training_priors['p_favorable_fill']:.6f}`")
    lines.append(f"- Training prior P(TOXIC): `{bundle.training_priors['p_toxic_fill']:.6f}`")
    lines.append(f"- Available post-trade windows (ms): `{bundle.available_post_trade_windows_ms}`")
    lines.append("")

    lines.append("## Economic Metric Notes")
    lines.append("")
    lines.append("- `Implementation Shortfall` is computed as signed execution price minus entry mid-price.")
    lines.append("- `Realized PnL` is computed as signed future mid-price minus execution price.")
    lines.append("- Passive fills use the dataset's post-trade window when available.")
    lines.append("- Immediate market entries and cleanup trades use the configured fallback future-mid mode.")
    lines.append("- Utility-based rejections are recorded as `cancel` with zero execution cost and zero realized PnL.")
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
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        values = [str(row[col]) for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _fmt_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"
