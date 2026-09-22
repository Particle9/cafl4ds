"""Summarize P1.2 comparison artifacts as a compact decision table.

Usage:
    .venv/Scripts/python scripts/summarize_adaptation_sweep.py outputs/adaptation-bdd/sweeps/<run>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def summarize(path: Path) -> dict[str, Any]:
    """Reduce one P1.2 comparison artifact to selection-relevant paired statistics."""
    report = json.loads(path.read_text(encoding="utf-8"))
    design = report.get("design", {})
    seeds = report["seeds"]
    knn = [float(seed["global"]["knn"]["gain"]) for seed in seeds]
    linear = [float(seed["global"]["linear"]["gain"]) for seed in seeds]
    final_rows = [seed["conditions"]["trajectory"][-1] for seed in seeds]
    current = [float(row["current_gain"]) for row in final_rows if row.get("current_gain") is not None]
    past = [float(row["past_gain_mean"]) for row in final_rows if row.get("past_gain_mean") is not None]
    joint_wins = sum(
        k > 0.0 and lin > 0.0 and row.get("past_gain_mean") is not None and float(row["past_gain_mean"]) >= 0.0
        for k, lin, row in zip(knn, linear, final_rows, strict=True)
    )
    return {
        "path": str(path.parent),
        "filter": design.get("selection", {}).get("filter", "unknown"),
        "init": design.get("init_mode", "unknown") + ("+well" if design.get("warm") else ""),
        "lr": float(design["lr"]),
        "cadence": int(design["update_every"]),
        "n": len(seeds),
        "knn_mean": _mean(knn),
        "knn_wins": sum(value > 0.0 for value in knn),
        "linear_mean": _mean(linear),
        "linear_wins": sum(value > 0.0 for value in linear),
        "current_mean": _mean(current),
        "past_mean": _mean(past),
        "retention_losses": sum(value < 0.0 for value in past),
        "joint_wins": joint_wins,
    }


def _fmt(value: object) -> str:
    return f"{value:+.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    """Find comparison artifacts recursively and print the P1.2 decision table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Sweep directory or one comparison.json")
    args = parser.parse_args()
    paths = [args.root] if args.root.is_file() else sorted(args.root.rglob("comparison.json"))
    if not paths:
        raise SystemExit(f"no comparison.json files found below {args.root}")
    rows = [summarize(path) for path in paths]
    rows.sort(key=lambda row: (row["retention_losses"], -row["joint_wins"], -row["linear_mean"]))
    columns = [
        "filter",
        "init",
        "lr",
        "cadence",
        "n",
        "knn_mean",
        "knn_wins",
        "linear_mean",
        "linear_wins",
        "current_mean",
        "past_mean",
        "retention_losses",
        "joint_wins",
        "path",
    ]
    widths = {column: max(len(column), *(len(_fmt(row[column])) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_fmt(row[column]).ljust(widths[column]) for column in columns))
    print()
    print("joint_wins requires positive kNN and linear gain plus non-negative earlier-condition gain in one seed.")
    if all(row["n"] == 5 for row in rows):
        print("Confirmation signal: look for >=4/5 joint wins and zero retention-loss seeds.")
    else:
        print("Development results select one cell only; they do not pass the five-seed confirmation gate.")


if __name__ == "__main__":
    main()
