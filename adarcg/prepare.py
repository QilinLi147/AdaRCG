"""Create the fixed AdaRCG cache format from released dataset features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SEED_TRIAL_LENGTHS = (
    235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206,
)
SEED_LABELS = (0, 1, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 2)
SEEDIV_LABELS = (
    (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
    (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
    (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
)
SEQUENCE_LENGTH = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_sequence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (62, 5):
        raise ValueError(f"trial feature shape must be [time,62,5], got {values.shape}")
    if len(values) < SEQUENCE_LENGTH:
        raise ValueError("trial is shorter than ten feature frames")
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=SEQUENCE_LENGTH, axis=0
    )
    return np.ascontiguousarray(
        windows.transpose(0, 1, 3, 2).reshape(len(windows), 62, 50)
    )


def empty_buffers(*, include_emotion: bool = False) -> dict[str, list[np.ndarray]]:
    names = ["x", "y", "subject", "session", "trial", "window_start"]
    if include_emotion:
        names.append("emotion")
    return {name: [] for name in names}


def append_trial(
    buffers: dict[str, list[np.ndarray]],
    values: np.ndarray,
    *,
    label: int,
    subject: int,
    session: int,
    trial: int,
    storage_dtype: np.dtype,
    emotion: int | None = None,
) -> None:
    windows = temporal_sequence(values).astype(storage_dtype)
    count = len(windows)
    buffers["x"].append(windows)
    buffers["y"].append(np.full(count, label, dtype=np.int8))
    buffers["subject"].append(np.full(count, subject, dtype=np.int8))
    buffers["session"].append(np.full(count, session, dtype=np.int8))
    buffers["trial"].append(np.full(count, trial, dtype=np.int16))
    buffers["window_start"].append(np.arange(count, dtype=np.int16))
    if emotion is not None:
        buffers["emotion"].append(np.full(count, emotion, dtype=np.int8))


def build_seed(root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    buffers = empty_buffers()
    for session in range(3):
        for subject in range(15):
            path = root / "DE" / "raw" / f"DE_{session * 15 + subject + 1}.mat"
            values = loadmat(path)["DE_feature"].transpose(1, 0, 2).astype(np.float32)
            if values.shape != (3394, 62, 5):
                raise ValueError(f"unexpected SEED feature shape in {path}: {values.shape}")
            start = 0
            for trial, (length, label) in enumerate(zip(SEED_TRIAL_LENGTHS, SEED_LABELS)):
                append_trial(
                    buffers,
                    values[start:start + length],
                    label=label,
                    subject=subject,
                    session=session,
                    trial=trial,
                    storage_dtype=np.float16,
                )
                start += length
    return _concatenate(buffers), {
        "dataset": "SEED",
        "subjects": 15,
        "sessions": 3,
        "trials_per_session": 15,
        "classes": 3,
        "partition_contract": "seed_t33_v1",
        "feature_source": "released unsmoothed one-second five-band DE",
        "storage_dtype": "float16",
    }


def build_mped(root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    three = loadmat(root / "mped_label_3type.mat")["DE_label"].reshape(-1).astype(np.int64)
    seven = loadmat(root / "mped_label.mat")["DE_label"].reshape(-1).astype(np.int64)
    if three.shape != (3360,) or seven.shape != (3360,):
        raise ValueError("MPED label files must contain 28 trials x 120 samples")
    if set(np.unique(three)) != {0, 1, 2} or set(np.unique(seven)) != set(range(7)):
        raise ValueError("unexpected MPED label IDs")
    buffers = empty_buffers(include_emotion=True)
    for subject in range(23):
        path = root / f"DE_{subject + 1}.mat"
        values = loadmat(path)["DE_feature"].astype(np.float32)
        if values.shape != (3360, 62, 5):
            raise ValueError(f"unexpected MPED feature shape in {path}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite MPED feature in {path}")
        for trial in range(28):
            start = trial * 120
            label = np.unique(three[start:start + 120])
            emotion = np.unique(seven[start:start + 120])
            if len(label) != 1 or len(emotion) != 1:
                raise ValueError("MPED trial labels must be constant")
            append_trial(
                buffers,
                values[start:start + 120],
                label=int(label[0]),
                subject=subject,
                session=0,
                trial=trial,
                storage_dtype=np.float32,
                emotion=int(emotion[0]),
            )
    return _concatenate(buffers), {
        "dataset": "MPED",
        "subjects": 23,
        "sessions": 1,
        "trials_per_session": 28,
        "classes": 3,
        "partition_contract": "mped_trial_disjoint_4fold_v1",
        "feature_source": "released artifact-cleaned one-second five-band features",
        "storage_dtype": "float32",
    }


def _seediv_trials(root: Path):
    for session in range(3):
        session_root = root / str(session + 1)
        for subject in range(15):
            matches = sorted(session_root.glob(f"{subject + 1}_*.mat"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one SEED-IV file for subject {subject + 1}, session {session + 1}"
                )
            archive = loadmat(matches[0])
            for trial, label in enumerate(SEEDIV_LABELS[session]):
                key = f"de_movingAve{trial + 1}"
                if key not in archive:
                    raise KeyError(f"{matches[0]} lacks {key}")
                yield (
                    np.asarray(archive[key], dtype=np.float32).transpose(1, 0, 2),
                    int(label),
                    subject,
                    session,
                    trial,
                )


def _seedv_trials(root: Path):
    for subject in range(16):
        path = root / f"{subject + 1}_123.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = pickle.loads(archive["data"].item())
            labels = pickle.loads(archive["label"].item())
        if sorted(data) != list(range(45)) or sorted(labels) != list(range(45)):
            raise ValueError(f"unexpected SEED-V trial keys in {path}")
        for key in range(45):
            session, trial = divmod(key, 15)
            values = np.asarray(data[key], dtype=np.float32).reshape(-1, 62, 5)
            label = np.unique(np.asarray(labels[key]).reshape(-1))
            if len(label) != 1:
                raise ValueError(f"non-constant SEED-V label in {path}, trial {key}")
            yield values, int(label[0]), subject, session, trial


def _apply_target_session_contract(arrays: dict[str, np.ndarray]) -> list[dict[str, int]]:
    session = arrays["session"].astype(np.int16)
    trial = arrays["trial"].astype(np.int16)
    label = arrays["y"].astype(np.int16)
    subject = arrays["subject"].astype(np.int16)
    audit: list[dict[str, int]] = []
    for subject_id in sorted(np.unique(subject).tolist()):
        subject_mask = subject == subject_id
        next_trial = int(np.max(trial[subject_mask & (session == 1)])) + 1
        for class_id in sorted(np.unique(label[subject_mask]).tolist()):
            target_trials = sorted(np.unique(
                trial[subject_mask & (session == 2) & (label == class_id)]
            ).tolist())
            if len(target_trials) < 3:
                raise ValueError("each class needs at least three target-session trials")
            original = int(target_trials[1])
            derived = next_trial + int(class_id)
            member = subject_mask & (session == 2) & (trial == original)
            session[member] = 1
            trial[member] = derived
            audit.append({
                "subject": int(subject_id),
                "class": int(class_id),
                "original_target_trial": original,
                "development_trial": derived,
                "samples": int(member.sum()),
            })
    arrays["session"] = session.astype(arrays["session"].dtype)
    arrays["trial"] = trial.astype(arrays["trial"].dtype)
    return audit


def build_seed_family(
    root: Path, dataset: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if dataset == "seediv":
        iterator = _seediv_trials(root)
        subjects, trials, classes = 15, 24, 4
        feature_source = "released de_movingAve five-band DE"
    else:
        iterator = _seedv_trials(root)
        subjects, trials, classes = 16, 15, 5
        feature_source = "released five-band DE"
    buffers = empty_buffers()
    count = 0
    for values, label, subject, session, trial in iterator:
        append_trial(
            buffers,
            values,
            label=label,
            subject=subject,
            session=session,
            trial=trial,
            storage_dtype=np.float16,
        )
        count += 1
    expected = subjects * 3 * trials
    if count != expected:
        raise AssertionError(f"expected {expected} trials, got {count}")
    arrays = _concatenate(buffers)
    audit = _apply_target_session_contract(arrays)
    return arrays, {
        "dataset": dataset.upper(),
        "subjects": subjects,
        "sessions": 3,
        "trials_per_session": trials,
        "classes": classes,
        "partition_contract": "balanced_target_session_development_v2",
        "partition_audit": audit,
        "feature_source": feature_source,
        "storage_dtype": "float16",
    }


def _concatenate(buffers: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}


def save_cache(
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    source: Path,
) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {output}")
    metadata = {
        **metadata,
        "schema": "adarcg_public_sequence_cache_v1",
        "channels": 62,
        "bands": 5,
        "sequence_length": SEQUENCE_LENGTH,
        "normalisation_applied": False,
        "source_directory": str(source.expanduser().resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, output)
    manifest = {
        "schema": "adarcg_public_cache_manifest_v1",
        "cache": str(output),
        "cache_sha256": sha256(output),
        "samples": int(len(arrays["y"])),
        "shape": list(arrays["x"].shape),
        "metadata": metadata,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("seed", "mped", "seediv", "seedv"), required=True)
    parser.add_argument("--source", type=Path, required=True, help="directory containing released features")
    parser.add_argument("--output", type=Path, required=True, help="destination .npz cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if args.dataset == "seed":
        arrays, metadata = build_seed(source)
    elif args.dataset == "mped":
        arrays, metadata = build_mped(source)
    else:
        arrays, metadata = build_seed_family(source, args.dataset)
    print(json.dumps(save_cache(args.output, arrays, metadata, source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
