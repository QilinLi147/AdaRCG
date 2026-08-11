"""Run AdaRCG on one subject or a complete public dataset protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import profile
from .data import dataset_folds, load_cache
from .engine import atomic_json, run_fold


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def aggregate(output: Path) -> dict[str, object]:
    records = []
    for path in sorted(output.glob("subject_*/fold_*/metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError("no completed fold metrics were found")
    subject_rows = []
    for subject in sorted({int(row["subject_one_based"]) for row in records}):
        members = [row for row in records if int(row["subject_one_based"]) == subject]
        subject_rows.append({
            "subject_one_based": subject,
            **{
                metric: float(np.mean([float(row[metric]) for row in members]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            },
        })
    summary = {
        "schema": "adarcg_public_aggregate_v1",
        "dataset": records[0]["dataset"],
        "task": records[0]["task"],
        "protocol": records[0]["protocol"],
        "completed_folds": len(records),
        "completed_subjects": len(subject_rows),
        "subject_metrics": subject_rows,
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = np.asarray([row[metric] for row in subject_rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    atomic_json(output / "aggregate.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("seed", "mped", "seediv", "seedv"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", type=int, help="one-based subject ID; default: all")
    parser.add_argument("--fold", type=int, help="one-based MPED fold; default: all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    args = parser.parse_args()
    current = profile(args.dataset)
    if args.subject is not None and not 1 <= args.subject <= current.subjects:
        parser.error(f"subject must be in [1,{current.subjects}]")
    if args.fold is not None and not 1 <= args.fold <= current.outer_folds:
        parser.error(f"fold must be in [1,{current.outer_folds}]")
    return args


def main() -> None:
    args = parse_args()
    current = profile(args.dataset)
    cache = args.cache.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = resolve_device(args.device)
    data = load_cache(cache, current)
    subjects = [args.subject - 1] if args.subject is not None else list(range(current.subjects))
    requested_fold = args.fold - 1 if args.fold is not None else None
    for subject in subjects:
        for fold in dataset_folds(data, current, subject):
            if requested_fold is not None and fold.outer_fold != requested_fold:
                continue
            fold_output = output / f"subject_{subject + 1:02d}" / f"fold_{fold.outer_fold + 1:02d}"
            metrics_path = fold_output / "metrics.json"
            if metrics_path.exists():
                print(f"skip completed {fold_output}", flush=True)
                continue
            metrics = run_fold(cache, data, fold, current, fold_output, device)
            print(json.dumps(metrics, sort_keys=True), flush=True)
    print(json.dumps(aggregate(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

