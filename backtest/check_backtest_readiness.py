#!/usr/bin/env python3
"""Validate whether the standardized static backtest inputs are ready.

This checker focuses on the current standardized static DeepHit workflow:

- artifacts are expected as ``*_base_net.pt`` / ``*_meta.pt`` pairs
- preprocessing is expected to match the notebook
  ``notebooks/baseline_models/standardized_static_deephit.ipynb``
- the canonical labeled dataset is the AAPL parquet used in that notebook

It does not run the backtest itself. Instead, it answers the practical question:
"If I point a generic backtester at these artifacts and data, what is still missing?"
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_PYTHON_MODULES = [
    "duckdb",
    "pyarrow",
    "torch",
    "pycox",
    "torchtuples",
]

OPTIONAL_MODEL_MODULES = {
    "mamba": "mamba_ssm",
}

STANDARDIZED_NOTEBOOK_REL = Path("notebooks/baseline_models/standardized_static_deephit.ipynb")
EXPECTED_DATASET_NAME = "labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101.parquet"
LEGACY_DATASET_NAMES = (
    "labeled_dataset_XNAS_ITCH_AAPL_mbo_20251001_20260101_equal.parquet",
)


@dataclass
class CheckResult:
    ok: bool
    message: str


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _read_torch_meta_without_torch(path: Path) -> dict[str, Any]:
    """Read a torch-saved metadata dict using stdlib tools only.

    ``torch.save`` stores data in a zip archive. The metadata files in this project
    contain plain Python / NumPy objects and can be read back from ``data.pkl``
    without importing PyTorch.
    """

    stem = path.stem
    with zipfile.ZipFile(path) as zf:
        data = zf.read(f"{stem}/data.pkl")
    obj = pickle.load(io.BytesIO(data))
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} did not contain a metadata dict.")
    return obj


def _check_parquet_footer(path: Path) -> CheckResult:
    try:
        with path.open("rb") as fh:
            head = fh.read(4)
            fh.seek(-4, 2)
            tail = fh.read(4)
    except OSError as exc:
        return CheckResult(False, f"{path.name}: unreadable ({exc})")

    if head != b"PAR1":
        return CheckResult(False, f"{path.name}: missing parquet header magic")
    if tail != b"PAR1":
        return CheckResult(False, f"{path.name}: missing parquet footer magic (likely truncated)")
    return CheckResult(True, f"{path.name}: footer/header look valid")


def _check_zstd_integrity(path: Path) -> CheckResult:
    proc = subprocess.run(
        ["zstd", "-t", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return CheckResult(True, f"{path.name}: zstd stream is valid")
    stderr = proc.stderr.strip().replace("\n", " ")
    return CheckResult(False, f"{path.name}: zstd integrity failed ({stderr})")


def _discover_artifact_pairs(artifacts_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(artifacts_dir.glob("*.pt")):
        name = path.name
        if name.endswith("_base_net.pt"):
            key = name[: -len("_base_net.pt")]
            pairs.setdefault(key, {})["base_net"] = path
        elif name.endswith("_meta.pt"):
            key = name[: -len("_meta.pt")]
            pairs.setdefault(key, {})["meta"] = path
    return pairs


def _load_notebook_dataset_hint(project_root: Path) -> CheckResult:
    notebook_path = project_root / STANDARDIZED_NOTEBOOK_REL
    if not notebook_path.exists():
        return CheckResult(False, f"Notebook not found: {notebook_path}")

    try:
        nb = json.loads(notebook_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive parsing
        return CheckResult(False, f"Failed to parse notebook JSON: {exc}")

    dataset_mentions = []
    legacy_mentions = set()
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if EXPECTED_DATASET_NAME in src:
            dataset_mentions.append(src)
        for legacy_name in LEGACY_DATASET_NAMES:
            if legacy_name in src:
                legacy_mentions.add(legacy_name)

    if not dataset_mentions:
        if legacy_mentions:
            legacy_list = ", ".join(sorted(legacy_mentions))
            return CheckResult(
                True,
                "Notebook still references legacy dataset name(s) "
                f"{legacy_list}; readiness now validates the current PSC dataset "
                f"{EXPECTED_DATASET_NAME}.",
            )
        return CheckResult(
            True,
            f"Notebook does not reference expected dataset {EXPECTED_DATASET_NAME}; "
            "using the current backtest CLI dataset contract instead.",
        )
    return CheckResult(
        True,
        "Notebook confirms the standardized static artifacts were trained against "
        f"{EXPECTED_DATASET_NAME}",
    )


def evaluate_readiness(project_root: Path, artifacts_dir: Path, data_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {
        "project_root": str(project_root),
        "artifacts_dir": str(artifacts_dir),
        "data_dir": str(data_dir),
        "environment": {},
        "artifacts": {},
        "data": {},
        "assumptions": [],
        "blocking_issues": [],
        "warnings": [],
    }

    for module_name in REQUIRED_PYTHON_MODULES:
        present = _has_module(module_name)
        results["environment"][module_name] = present
        if not present:
            results["blocking_issues"].append(
                f"Missing Python dependency: {module_name}"
            )

    pairs = _discover_artifact_pairs(artifacts_dir)
    if not pairs:
        results["blocking_issues"].append(
            f"No *.pt artifact pairs found in {artifacts_dir}"
        )
    else:
        for key, pair in pairs.items():
            entry: dict[str, Any] = {
                "base_net_present": "base_net" in pair,
                "meta_present": "meta" in pair,
            }
            if "meta" in pair:
                try:
                    meta = _read_torch_meta_without_torch(pair["meta"])
                    entry["meta"] = {
                        "model_name": meta.get("model_name"),
                        "num_competing_events": meta.get("num_competing_events"),
                        "output_steps": meta.get("output_steps"),
                        "lookback_steps": meta.get("lookback_steps"),
                    }
                    model_name = meta.get("model_name")
                    optional_dep = OPTIONAL_MODEL_MODULES.get(model_name)
                    if optional_dep and not _has_module(optional_dep):
                        results["blocking_issues"].append(
                            f"Artifact {key} requires optional dependency {optional_dep}"
                        )
                except Exception as exc:
                    entry["meta_error"] = str(exc)
                    results["blocking_issues"].append(
                        f"Failed to read artifact metadata for {key}: {exc}"
                    )

            if "base_net" not in pair or "meta" not in pair:
                results["blocking_issues"].append(
                    f"Artifact pair incomplete for {key} (need both *_base_net.pt and *_meta.pt)"
                )
            results["artifacts"][key] = entry

    notebook_check = _load_notebook_dataset_hint(project_root)
    results["data"]["notebook_dataset_match"] = {
        "ok": notebook_check.ok,
        "message": notebook_check.message,
    }
    if not notebook_check.ok:
        results["blocking_issues"].append(notebook_check.message)
    else:
        notebook_hint_is_warning = (
            "legacy dataset" in notebook_check.message
            or "does not reference expected dataset" in notebook_check.message
        )
        if notebook_hint_is_warning:
            results["warnings"].append(notebook_check.message)

    labeled_dataset = data_dir / "datasets" / EXPECTED_DATASET_NAME
    if not labeled_dataset.exists():
        results["blocking_issues"].append(
            f"Canonical labeled dataset is missing: {labeled_dataset}"
        )
    else:
        parquet_check = _check_parquet_footer(labeled_dataset)
        results["data"]["canonical_labeled_dataset"] = {
            "ok": parquet_check.ok,
            "message": parquet_check.message,
        }
        if not parquet_check.ok:
            results["blocking_issues"].append(parquet_check.message)

    raw_checks: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("raw/*/*.dbn.zst"))[:4]:
        check = _check_zstd_integrity(path)
        raw_checks.append({"path": str(path), "ok": check.ok, "message": check.message})
        if not check.ok:
            results["blocking_issues"].append(check.message)
    results["data"]["raw_samples"] = raw_checks

    parquet_samples: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("datasets/*.parquet"))[:4]:
        check = _check_parquet_footer(path)
        parquet_samples.append({"path": str(path), "ok": check.ok, "message": check.message})
        if not check.ok:
            results["blocking_issues"].append(check.message)
    results["data"]["dataset_samples"] = parquet_samples

    results["assumptions"].append(
        "The standardized static artifacts do not persist feature mean/std, so any "
        "generic backtester must either reproduce the original train split exactly "
        "or extend the artifact format to save preprocessing statistics."
    )
    results["assumptions"].append(
        "The current repository includes reusable static backtest CLIs: "
        "`python -m backtest.run_backtest` and `python -m backtest.threshold_sweep`; "
        "this checker validates artifacts, datasets, and dependencies before use."
    )

    overall_ready = len(results["blocking_issues"]) == 0
    results["overall_ready"] = overall_ready
    return results


def print_human_summary(report: dict[str, Any]) -> None:
    print("Backtest Readiness Report")
    print("=" * 80)
    print(f"Project root : {report['project_root']}")
    print(f"Artifacts dir: {report['artifacts_dir']}")
    print(f"Data dir     : {report['data_dir']}")
    print()

    print("Environment")
    for name, present in report["environment"].items():
        status = "OK" if present else "MISSING"
        print(f"  [{status:<7}] {name}")
    print()

    print("Artifacts")
    for name, entry in sorted(report["artifacts"].items()):
        print(f"  - {name}")
        print(f"      base_net: {entry.get('base_net_present')}")
        print(f"      meta    : {entry.get('meta_present')}")
        meta = entry.get("meta")
        if meta:
            print(
                "      meta    : "
                f"model={meta.get('model_name')} "
                f"K={meta.get('num_competing_events')} "
                f"T={meta.get('output_steps')} "
                f"lookback={meta.get('lookback_steps')}"
            )
        if entry.get("meta_error"):
            print(f"      error   : {entry['meta_error']}")
    print()

    print("Data checks")
    notebook = report["data"].get("notebook_dataset_match")
    if notebook:
        status = "OK" if notebook["ok"] else "FAIL"
        print(f"  [{status:<4}] notebook alignment: {notebook['message']}")
    labeled = report["data"].get("canonical_labeled_dataset")
    if labeled:
        status = "OK" if labeled["ok"] else "FAIL"
        print(f"  [{status:<4}] canonical labeled dataset: {labeled['message']}")
    for entry in report["data"].get("dataset_samples", []):
        status = "OK" if entry["ok"] else "FAIL"
        print(f"  [{status:<4}] parquet sample: {entry['message']}")
    for entry in report["data"].get("raw_samples", []):
        status = "OK" if entry["ok"] else "FAIL"
        print(f"  [{status:<4}] raw sample: {entry['message']}")
    print()

    print("Assumptions / warnings")
    for item in report["assumptions"]:
        print(f"  - {item}")
    for item in report["warnings"]:
        print(f"  - WARNING: {item}")
    print()

    if report["blocking_issues"]:
        print("Blocking issues")
        for item in report["blocking_issues"]:
            print(f"  - {item}")
    else:
        print("Blocking issues")
        print("  - None")
    print()

    print(f"Overall ready: {report['overall_ready']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = Path("/ocean/projects/cis260122p/hwang71/lob-deep-survival-analysis-main")
    default_artifacts = Path("/ocean/projects/cis260122p/shared/artifacts/baseline")
    default_data = Path("/ocean/projects/cis260122p/shared/data")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = evaluate_readiness(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        data_dir=args.data_dir,
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_human_summary(report)
    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
