#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd


MODEL_ORDER = ["gru", "transformer", "gru_transformer", "mamba"]
MODEL_LABELS = {
    "gru": "GRU",
    "transformer": "Transformer",
    "gru_transformer": "GRU-Transformer",
    "mamba": "Mamba",
}

MODEL_COLORS = {
    "gru": "#4C78A8",
    "transformer": "#F58518",
    "gru_transformer": "#54A24B",
    "mamba": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready backtest figures from aggregated CSV outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("backtest/outputs/retest_threshold0_static"),
        help="Directory containing combined_summary.csv and model_vs_baselines.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory where report figures and helper CSV should be written",
    )
    return parser.parse_args()


def load_inputs(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.read_csv(input_root / "combined_summary.csv")
    comparison = pd.read_csv(input_root / "model_vs_baselines.csv")
    return combined, comparison


def prepare_model_rows(combined: pd.DataFrame) -> pd.DataFrame:
    model_rows = combined.loc[combined["strategy"] == "model"].copy()
    model_rows["model"] = pd.Categorical(
        model_rows["model"], categories=MODEL_ORDER, ordered=True
    )
    model_rows = model_rows.sort_values(["model", "ticker"]).reset_index(drop=True)
    return model_rows


def export_model_level_means(model_rows: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = [
        "mean_implementation_shortfall",
        "mean_realized_pnl",
        "toxic_fill_rate_overall",
        "toxic_fill_rate_among_passive",
        "limit_decision_rate",
        "passive_fill_rate",
        "cleanup_rate",
        "market_entry_rate",
        "cancel_rate",
    ]
    means = (
        model_rows.groupby("model", observed=True)[cols]
        .mean()
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    means["model_label"] = means["model"].map(MODEL_LABELS)
    means.to_csv(output_dir / "backtest_model_level_means.csv", index=False)
    return means


def _set_common_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_headline_panels(model_rows: pd.DataFrame, output_dir: Path) -> None:
    _set_common_style()
    metrics = [
        ("mean_implementation_shortfall", "Implementation Shortfall"),
        ("mean_realized_pnl", "Realized PnL"),
        ("toxic_fill_rate_overall", "Toxic Fill Rate"),
    ]

    means = (
        model_rows.groupby("model", observed=True)[[m for m, _ in metrics]]
        .mean()
        .reindex(MODEL_ORDER)
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    x = np.arange(len(MODEL_ORDER))

    for ax, (metric, title) in zip(axes, metrics):
        bar_colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        ax.bar(x, means[metric].values, color=bar_colors, alpha=0.85, width=0.68)

        for idx, model in enumerate(MODEL_ORDER):
            values = (
                model_rows.loc[model_rows["model"] == model, metric]
                .astype(float)
                .to_numpy()
            )
            if values.size:
                jitter = np.linspace(-0.12, 0.12, values.size)
                ax.scatter(
                    np.full(values.size, idx) + jitter,
                    values,
                    color="black",
                    s=14,
                    alpha=0.7,
                    linewidths=0,
                    zorder=3,
                )

        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "backtest_economic_headline_panels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _heatmap_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
) -> None:
    arr = data.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(arr, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isnan(value):
                text = "NA"
                color = "black"
            else:
                text = f"{value:.3f}"
                color = "white" if abs(value) > 0.45 * vmax else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    return im


def make_always_limit_improvement_heatmap(comparison: pd.DataFrame, output_dir: Path) -> None:
    _set_common_style()
    base = comparison.copy()
    base["model"] = pd.Categorical(base["model"], categories=MODEL_ORDER, ordered=True)
    base = base.sort_values(["ticker", "model"]).reset_index(drop=True)

    improvement_specs = [
        (
            "is_improvement",
            -base["delta_mean_implementation_shortfall__model_minus_always_limit"],
            "IS Improvement vs Always Limit",
        ),
        (
            "pnl_improvement",
            base["delta_mean_realized_pnl__model_minus_always_limit"],
            "PnL Improvement vs Always Limit",
        ),
        (
            "toxic_improvement",
            -base["delta_toxic_fill_rate_overall__model_minus_always_limit"],
            "Toxic Fill Improvement vs Always Limit",
        ),
    ]

    tickers = sorted(base["ticker"].unique())
    model_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.2))

    for ax, (name, series, title) in zip(axes, improvement_specs):
        heat_df = base[["ticker", "model"]].copy()
        heat_df[name] = series.to_numpy()
        pivot = (
            heat_df.pivot(index="ticker", columns="model", values=name)
            .reindex(index=tickers, columns=MODEL_ORDER)
        )
        im = _heatmap_panel(ax, pivot, tickers, model_labels, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(
        output_dir / "backtest_vs_always_limit_improvement_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_execution_behavior_heatmap(model_level_means: pd.DataFrame, output_dir: Path) -> None:
    _set_common_style()
    metrics = [
        "limit_decision_rate",
        "passive_fill_rate",
        "cleanup_rate",
        "market_entry_rate",
        "cancel_rate",
    ]

    heat = (
        model_level_means.set_index("model")
        .reindex(MODEL_ORDER)[metrics]
        .astype(float)
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(heat.to_numpy(), cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("Execution Behavior by Model")

    arr = heat.to_numpy(dtype=float)
    vmax = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 1.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if value > 0.55 * vmax else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        output_dir / "backtest_execution_behavior_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined, comparison = load_inputs(args.input_root)
    model_rows = prepare_model_rows(combined)
    model_level_means = export_model_level_means(model_rows, args.output_dir)

    make_headline_panels(model_rows, args.output_dir)
    make_always_limit_improvement_heatmap(comparison, args.output_dir)
    make_execution_behavior_heatmap(model_level_means, args.output_dir)

    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
