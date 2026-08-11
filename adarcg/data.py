"""Cache validation, fixed dataset partitions and train-fitted normalisation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import DatasetProfile


@dataclass(frozen=True)
class Fold:
    subject: int
    outer_fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Normalisation:
    mean: np.ndarray
    std: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path, profile: DatasetProfile) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "subject", "session", "trial", "metadata"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"cache is missing fields: {sorted(missing)}")
        data = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(data.pop("metadata")))
    data["metadata"] = metadata
    data["x"] = np.asarray(data["x"], dtype=np.float32)
    data["y"] = np.asarray(data["y"], dtype=np.int64)
    for name in ("subject", "session", "trial"):
        data[name] = np.asarray(data[name], dtype=np.int64)
    sample_count = len(data["y"])
    if any(len(data[name]) != sample_count for name in ("x", "subject", "session", "trial")):
        raise ValueError("cache arrays do not have a common sample count")
    if data["x"].ndim != 3 or data["x"].shape[1:] != (62, 50):
        raise ValueError(f"expected x with shape [sample,62,50], got {data['x'].shape}")
    if not np.isfinite(data["x"]).all():
        raise ValueError("cache contains non-finite EEG features")
    if sorted(np.unique(data["y"]).tolist()) != list(range(profile.classes)):
        raise ValueError("cache class IDs do not match the fixed dataset profile")
    subjects = sorted(np.unique(data["subject"]).tolist())
    if subjects != list(range(profile.subjects)):
        raise ValueError("cache subject IDs do not match the fixed dataset profile")
    if profile.name == "mped" and "emotion" not in data:
        raise ValueError("MPED cache must include the original seven-emotion trial IDs")
    if profile.name in {"seediv", "seedv"} and metadata.get("partition_contract") != (
        "balanced_target_session_development_v2"
    ):
        raise ValueError(
            f"{profile.name.upper()} cache was not built with the public fixed partition contract"
        )
    return data


def _trial_label(data: dict[str, np.ndarray], subject: int, session: int, trial: int) -> int:
    member = (
        (data["subject"] == subject)
        & (data["session"] == session)
        & (data["trial"] == trial)
    )
    labels = np.unique(data["y"][member])
    if len(labels) != 1:
        raise ValueError("each trial must have one label")
    return int(labels[0])


def _seed_fold(data: dict[str, np.ndarray], subject: int) -> Fold:
    subject_mask = data["subject"] == subject
    session, trial = data["session"], data["trial"]
    train = np.flatnonzero(subject_mask & (
        (session == 0)
        | ((session == 1) & (trial < 12))
        | ((session == 2) & (trial < 3))
    ))
    validation = np.flatnonzero(subject_mask & (session == 1) & (trial >= 12))
    test = np.flatnonzero(subject_mask & (session == 2) & (trial >= 3))
    return _validate_fold(Fold(subject, 0, train, validation, test))


def _seed_family_fold(data: dict[str, np.ndarray], subject: int) -> Fold:
    subject_mask = data["subject"] == subject
    classes = list(range(len(np.unique(data["y"][subject_mask]))))
    labels_by_session: dict[int, dict[int, int]] = {}
    for session in range(3):
        trials = sorted(np.unique(data["trial"][subject_mask & (data["session"] == session)]))
        labels_by_session[session] = {
            int(trial): _trial_label(data, subject, session, int(trial)) for trial in trials
        }
    development = {
        max(trial for trial, label in labels_by_session[1].items() if label == class_id)
        for class_id in classes
    }
    calibration = {
        min(trial for trial, label in labels_by_session[2].items() if label == class_id)
        for class_id in classes
    }
    session, trial = data["session"], data["trial"]
    train = np.flatnonzero(subject_mask & (
        (session == 0)
        | ((session == 1) & ~np.isin(trial, sorted(development)))
        | ((session == 2) & np.isin(trial, sorted(calibration)))
    ))
    validation = np.flatnonzero(
        subject_mask & (session == 1) & np.isin(trial, sorted(development))
    )
    test = np.flatnonzero(
        subject_mask & (session == 2) & ~np.isin(trial, sorted(calibration))
    )
    return _validate_fold(Fold(subject, 0, train, validation, test))


def _mped_folds(data: dict[str, np.ndarray], subject: int) -> list[Fold]:
    subject_mask = data["subject"] == subject
    trial_emotion: dict[int, int] = {}
    for trial in np.unique(data["trial"][subject_mask]):
        values = np.unique(data["emotion"][subject_mask & (data["trial"] == trial)])
        if len(values) != 1:
            raise ValueError("each MPED trial must have one seven-emotion ID")
        trial_emotion[int(trial)] = int(values[0])
    result = []
    for outer_fold in range(4):
        trial_sets = {"train": [], "validation": [], "test": []}
        for emotion in range(7):
            group = sorted(trial for trial, label in trial_emotion.items() if label == emotion)
            if len(group) != 4:
                raise ValueError("MPED requires four trials per original emotion")
            trial_sets["test"].append(group[outer_fold])
            trial_sets["validation"].append(group[(outer_fold + 1) % 4])
            trial_sets["train"].extend(
                trial for trial in group
                if trial not in {group[outer_fold], group[(outer_fold + 1) % 4]}
            )
        index = {
            name: np.flatnonzero(subject_mask & np.isin(data["trial"], trials))
            for name, trials in trial_sets.items()
        }
        result.append(_validate_fold(Fold(
            subject, outer_fold, index["train"], index["validation"], index["test"]
        )))
    return result


def _validate_fold(fold: Fold) -> Fold:
    parts = (fold.train, fold.validation, fold.test)
    if any(len(part) == 0 for part in parts):
        raise ValueError("fixed protocol produced an empty partition")
    combined = np.concatenate(parts)
    if len(np.unique(combined)) != len(combined):
        raise AssertionError("sample leakage across train, development and test partitions")
    return fold


def dataset_folds(data: dict[str, np.ndarray], profile: DatasetProfile, subject: int) -> list[Fold]:
    if not 0 <= subject < profile.subjects:
        raise ValueError(f"subject must be in [0,{profile.subjects - 1}]")
    if profile.name == "seed":
        return [_seed_fold(data, subject)]
    if profile.name in {"seediv", "seedv"}:
        return [_seed_family_fold(data, subject)]
    return _mped_folds(data, subject)


def fit_normalisation(inputs: np.ndarray, indices: np.ndarray) -> Normalisation:
    selected = inputs[np.asarray(indices, dtype=np.int64)].reshape(len(indices), 62, 10, 5)
    mean = selected.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = selected.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    return Normalisation(mean=mean, std=np.maximum(std, np.float32(1e-6)))


class CacheDataset(Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        indices: np.ndarray,
        normalisation: Normalisation,
    ) -> None:
        self.data = data
        self.x = data["x"]
        self.y = data["y"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalisation = normalisation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        sequence = self.x[index].reshape(62, 10, 5)
        sequence = (sequence - self.normalisation.mean[:, None, :]) / (
            self.normalisation.std[:, None, :]
        )
        return (
            torch.from_numpy(np.ascontiguousarray(sequence.reshape(62, 50), dtype=np.float32)),
            torch.tensor(int(self.y[index])),
            torch.tensor(int(self.data["trial"][index])),
            torch.tensor(int(self.data["subject"][index])),
            torch.tensor(int(self.data["session"][index])),
        )


def make_loader(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    normalisation: Normalisation,
    profile: DatasetProfile,
    *,
    shuffle: bool,
    seed: int,
    evaluation: bool = False,
) -> DataLoader:
    dataset = CacheDataset(data, indices, normalisation)
    batch_size = profile.eval_batch_size if evaluation else profile.batch_size
    generator = torch.Generator().manual_seed(int(seed))
    if shuffle and profile.session_weight != 1.0:
        weights = np.where(
            data["session"][indices] == 2, profile.session_weight, 1.0
        ).astype(np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), len(indices), replacement=True, generator=generator
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )

