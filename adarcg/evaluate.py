"""Replay a public AdaRCG checkpoint on its stored outer-test partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import profile
from .data import Normalisation, file_sha256, load_cache, make_loader
from .engine import atomic_predictions, load_checkpoint, metric_set, predict
from .run import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--output", type=Path, help="optional prediction .npz path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    cache_path = args.cache.expanduser().resolve()
    device = resolve_device(args.device)
    model, payload = load_checkpoint(checkpoint_path, device)
    current = profile(str(payload["dataset"]))
    if file_sha256(cache_path) != payload["cache_sha256"]:
        raise ValueError("cache fingerprint differs from the checkpoint record")
    data = load_cache(cache_path, current)
    normalisation = Normalisation(
        mean=np.asarray(payload["normalisation"]["mean"], dtype=np.float32),
        std=np.asarray(payload["normalisation"]["std"], dtype=np.float32),
    )
    test_indices = np.asarray(payload["split"]["outer_test"], dtype=np.int64)
    loader = make_loader(
        data, test_indices, normalisation, current,
        shuffle=False, seed=int(payload["fixed_seed"]), evaluation=True,
    )
    report = predict(model, loader, device, current)
    predictions = report["probability"].argmax(axis=1).astype(np.int64)
    scores = metric_set(report["label"], predictions)
    result = {
        "dataset": current.name,
        "subject_one_based": int(payload["subject_zero_based"]) + 1,
        "outer_fold_one_based": int(payload["outer_fold_zero_based"]) + 1,
        **scores,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_predictions(
            output,
            sample_index=test_indices,
            true_label=report["label"],
            prediction=predictions,
            class_probability=report["probability"].astype(np.float32),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
