"""Fixed paper configurations used by the public reproduction commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    task: str
    protocol: str
    subjects: int
    outer_folds: int
    classes: int
    learning_rate: float
    dropout: float
    max_epochs: int
    patience: int
    causal_decay: float
    anchor_c: float
    session_weight: float
    class_balance: str
    fusion_cap: float | None
    quality_power: float
    batch_size: int = 64
    eval_batch_size: int = 128
    gradient_accumulation_steps: int = 8
    hidden_dim: int = 48
    graph_rank: int = 4
    selected_regions: int = 2


PROFILES = {
    "seed": DatasetProfile(
        name="seed",
        task="emotion",
        protocol="seed_t33_v1",
        subjects=15,
        outer_folds=1,
        classes=3,
        learning_rate=1e-3,
        dropout=0.15,
        max_epochs=5,
        patience=2,
        causal_decay=0.95,
        anchor_c=0.2,
        session_weight=20.0,
        class_balance="none",
        fusion_cap=None,
        quality_power=1.0,
    ),
    "mped": DatasetProfile(
        name="mped",
        task="three",
        protocol="mped_trial_disjoint_4fold_v1",
        subjects=23,
        outer_folds=4,
        classes=3,
        learning_rate=1e-3,
        dropout=0.15,
        max_epochs=5,
        patience=2,
        causal_decay=0.90,
        anchor_c=0.2,
        session_weight=1.0,
        class_balance="none",
        fusion_cap=None,
        quality_power=1.0,
        gradient_accumulation_steps=2,
    ),
    "seediv": DatasetProfile(
        name="seediv",
        task="emotion",
        protocol="seediv_balanced_target_session_development_v2",
        subjects=15,
        outer_folds=1,
        classes=4,
        learning_rate=1e-3,
        dropout=0.15,
        max_epochs=60,
        patience=10,
        causal_decay=0.90,
        anchor_c=1.0,
        session_weight=40.0,
        class_balance="sqrt_inverse",
        fusion_cap=0.8,
        quality_power=2.0,
    ),
    "seedv": DatasetProfile(
        name="seedv",
        task="emotion",
        protocol="seedv_balanced_target_session_development_v2",
        subjects=16,
        outer_folds=1,
        classes=5,
        learning_rate=8e-4,
        dropout=0.30,
        max_epochs=40,
        patience=8,
        causal_decay=0.99,
        anchor_c=0.5,
        session_weight=40.0,
        class_balance="none",
        fusion_cap=0.2,
        quality_power=0.5,
    ),
}


def profile(dataset: str) -> DatasetProfile:
    key = dataset.lower().replace("-", "")
    if key not in PROFILES:
        raise ValueError(f"unsupported dataset: {dataset}")
    return PROFILES[key]

