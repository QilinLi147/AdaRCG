"""Training, replay and evaluation for the fixed public AdaRCG protocols."""

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import DatasetProfile
from .data import (
    Fold,
    Normalisation,
    file_sha256,
    fit_normalisation,
    make_loader,
)
from .model import AdaRCG
from .regions import dataset_regions


FIXED_SEED = 8635


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_predictions(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_everything(seed: int = FIXED_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(profile: DatasetProfile, device: torch.device) -> AdaRCG:
    membership, adjacency = dataset_regions(profile.name)
    return AdaRCG(
        torch.from_numpy(membership),
        torch.from_numpy(adjacency),
        input_dim=50,
        classes=profile.classes,
        hidden_dim=profile.hidden_dim,
        rank=profile.graph_rank,
        top_regions=profile.selected_regions,
        dropout=profile.dropout,
        auxiliary_weight=0.0,
        fusion_mode="reliability",
        structured_evidence=True,
    ).to(device)


def model_config(profile: DatasetProfile) -> dict[str, object]:
    return {
        "input_dim": 50,
        "classes": profile.classes,
        "hidden_dim": profile.hidden_dim,
        "rank": profile.graph_rank,
        "top_regions": profile.selected_regions,
        "dropout": profile.dropout,
        "auxiliary_weight": 0.0,
        "fusion_mode": "reliability",
        "structured_evidence": True,
    }


def decision_logits(
    model: AdaRCG,
    outputs: dict[str, torch.Tensor],
    profile: DatasetProfile,
) -> torch.Tensor:
    if profile.fusion_cap is None:
        return model.combined_logits(outputs)
    anchor = outputs["stable_logits"] + outputs["connection_logits"]
    quality = outputs["reliability"].mean(dim=1).clamp(1e-4, 1.0)
    graph_weight = (
        profile.fusion_cap
        * torch.sigmoid(model.decision_fusion_logits)
        * quality.pow(profile.quality_power)
    )
    return anchor + graph_weight[:, None] * torch.tanh(outputs["logits"])


def causal_integrate(
    probabilities: torch.Tensor,
    keys: list[tuple[int, int]],
    decay: float,
) -> torch.Tensor:
    states: list[torch.Tensor] = []
    state: torch.Tensor | None = None
    previous: tuple[int, int] | None = None
    for current, key in zip(probabilities, keys):
        state = current if key != previous else decay * state + (1.0 - decay) * current
        states.append(state)
        previous = key
    return torch.stack(states) if states else probabilities.clone()


@torch.no_grad()
def predict(
    model: AdaRCG,
    loader: DataLoader,
    device: torch.device,
    profile: DatasetProfile,
) -> dict[str, np.ndarray]:
    model.eval()
    logits, labels, trials, subjects, sessions = [], [], [], [], []
    for inputs, label, trial, subject, session in loader:
        outputs = model(inputs.to(device, non_blocking=True))
        logits.append(decision_logits(model, outputs, profile).cpu())
        labels.append(label.numpy())
        trials.append(trial.numpy())
        subjects.append(subject.numpy())
        sessions.append(session.numpy())
    instantaneous = torch.cat(logits, dim=0)
    trial_array = np.concatenate(trials).astype(np.int64)
    session_array = np.concatenate(sessions).astype(np.int64)
    keys = list(zip(session_array.tolist(), trial_array.tolist()))
    probabilities = causal_integrate(
        torch.softmax(instantaneous, dim=1), keys, profile.causal_decay
    )
    return {
        "logits": probabilities.clamp_min(1e-8).log().numpy(),
        "probability": probabilities.numpy(),
        "label": np.concatenate(labels).astype(np.int64),
        "trial": trial_array,
        "subject": np.concatenate(subjects).astype(np.int64),
        "session": session_array,
    }


def metric_set(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": 100.0 * accuracy_score(labels, predictions),
        "balanced_accuracy": 100.0 * balanced_accuracy_score(labels, predictions),
        "macro_f1": 100.0 * f1_score(labels, predictions, average="macro", zero_division=0),
    }


def initialise_anchor(
    model: AdaRCG,
    loader: DataLoader,
    normalisation: Normalisation,
    profile: DatasetProfile,
    device: torch.device,
) -> None:
    model.evidence_input_mean.copy_(torch.as_tensor(normalisation.mean, device=device))
    model.evidence_input_scale.copy_(torch.as_tensor(normalisation.std, device=device))
    ordered = DataLoader(
        loader.dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    blocks, labels, sessions = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, label, _trial, _subject, session in ordered:
            blocks.append(model.structured_evidence_features(inputs.to(device)).cpu().numpy())
            labels.append(label.numpy())
            sessions.append(session.numpy())
    features = np.concatenate(blocks).astype(np.float64, copy=False)
    label_array = np.concatenate(labels).astype(np.int64, copy=False)
    session_array = np.concatenate(sessions).astype(np.int64, copy=False)
    sample_weight = np.where(session_array == 2, profile.session_weight, 1.0)
    scaler = StandardScaler().fit(features, sample_weight=sample_weight)
    transformed = scaler.transform(features)
    classifier = LogisticRegression(
        C=profile.anchor_c,
        max_iter=400,
        class_weight="balanced",
        random_state=FIXED_SEED,
    ).fit(transformed, label_array, sample_weight=sample_weight)
    coefficient = np.asarray(classifier.coef_, dtype=np.float32)
    intercept = np.asarray(classifier.intercept_, dtype=np.float32)
    if profile.classes == 2 and coefficient.shape[0] == 1:
        coefficient = np.concatenate((-0.5 * coefficient, 0.5 * coefficient))
        intercept = np.concatenate((-0.5 * intercept, 0.5 * intercept))
    model.set_structured_evidence_state(
        input_mean=torch.as_tensor(normalisation.mean, device=device),
        input_scale=torch.as_tensor(normalisation.std, device=device),
        feature_mean=torch.as_tensor(scaler.mean_, device=device),
        feature_scale=torch.as_tensor(scaler.scale_, device=device),
        coefficient=torch.as_tensor(coefficient, device=device),
        intercept=torch.as_tensor(intercept, device=device),
    )
    for module in (model.stable_head, model.connection_evidence_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.train()


def class_weights(
    loader: DataLoader,
    profile: DatasetProfile,
    device: torch.device,
) -> torch.Tensor | None:
    if profile.class_balance == "none":
        return None
    dataset = loader.dataset
    counts = np.bincount(
        dataset.y[dataset.indices], minlength=profile.classes
    ).astype(np.float32)
    exponent = -0.5 if profile.class_balance == "sqrt_inverse" else -1.0
    weights = np.maximum(counts, 1.0) ** exponent
    weights /= weights.mean()
    return torch.from_numpy(weights).to(device)


def train_stage(
    model: AdaRCG,
    train_loader: DataLoader,
    report_loader: DataLoader,
    profile: DatasetProfile,
    device: torch.device,
    *,
    epochs: int,
    select_epoch: bool,
    stage: str,
    history: list[dict[str, object]],
) -> tuple[dict[str, torch.Tensor], int]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=profile.learning_rate,
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=profile.learning_rate * 0.05
    )
    weights = class_weights(train_loader, profile, device)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, (inputs, labels, _trial, _subject, _session) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = decision_logits(model, model(inputs), profile)
            loss = F.cross_entropy(
                logits, labels, weight=weights, label_smoothing=0.04
            )
            (loss / profile.gradient_accumulation_steps).backward()
            if (
                (batch_index + 1) % profile.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        scheduler.step()
        report = predict(model, report_loader, device, profile)
        scores = metric_set(report["label"], report["probability"].argmax(axis=1))
        improved = scores["balanced_accuracy"] > best_score + 1e-5
        if improved or not select_epoch:
            best_score = scores["balanced_accuracy"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0 if improved else stale + 1
        else:
            stale += 1
        history.append({
            "stage": stage,
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            **scores,
            "selected": 0,
        })
        print(
            f"{stage} epoch={epoch + 1:03d} "
            f"development_bacc={scores['balanced_accuracy']:.2f}",
            flush=True,
        )
        if select_epoch and stale >= profile.patience:
            break
    for row in history:
        if row["stage"] == stage and int(row["epoch"]) == best_epoch:
            row["selected"] = 1
    return best_state, best_epoch


def checkpoint_model(
    checkpoint: dict[str, object], device: torch.device
) -> AdaRCG:
    state = checkpoint["state_dict"]
    config = checkpoint["model_config"]
    model = AdaRCG(state["membership"], state["region_prior"], **config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[AdaRCG, dict[str, object]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "adarcg_public_checkpoint_v1":
        raise ValueError("unsupported checkpoint schema")
    return checkpoint_model(checkpoint, device), checkpoint


def run_fold(
    cache: Path,
    data: dict[str, np.ndarray],
    fold: Fold,
    profile: DatasetProfile,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = ("checkpoint.pt", "metrics.json", "predictions.npz")
    if any((output / name).exists() for name in protected):
        raise FileExistsError(f"refusing to overwrite completed fold: {output}")
    selection_norm = fit_normalisation(data["x"], fold.train)
    selection_train = make_loader(
        data, fold.train, selection_norm, profile, shuffle=True, seed=FIXED_SEED
    )
    selection_report = make_loader(
        data, fold.validation, selection_norm, profile,
        shuffle=False, seed=FIXED_SEED, evaluation=True,
    )
    history: list[dict[str, object]] = []
    seed_everything()
    model = build_model(profile, device)
    initialise_anchor(model, selection_train, selection_norm, profile, device)
    _selection_state, selected_epoch = train_stage(
        model,
        selection_train,
        selection_report,
        profile,
        device,
        epochs=profile.max_epochs,
        select_epoch=True,
        stage="selection",
        history=history,
    )
    del model, _selection_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_indices = np.concatenate((fold.train, fold.validation))
    refit_norm = fit_normalisation(data["x"], refit_indices)
    refit_train = make_loader(
        data, refit_indices, refit_norm, profile, shuffle=True, seed=FIXED_SEED
    )
    refit_report = make_loader(
        data, refit_indices, refit_norm, profile,
        shuffle=False, seed=FIXED_SEED, evaluation=True,
    )
    seed_everything()
    model = build_model(profile, device)
    initialise_anchor(model, refit_train, refit_norm, profile, device)
    refit_state, refit_epoch = train_stage(
        model,
        refit_train,
        refit_report,
        profile,
        device,
        epochs=selected_epoch,
        select_epoch=False,
        stage="refit",
        history=history,
    )
    if refit_epoch != selected_epoch:
        raise AssertionError("refit did not complete the selected number of epochs")
    model.load_state_dict(refit_state, strict=True)
    checkpoint = {
        "schema": "adarcg_public_checkpoint_v1",
        "dataset": profile.name,
        "task": profile.task,
        "protocol": profile.protocol,
        "subject_zero_based": fold.subject,
        "outer_fold_zero_based": fold.outer_fold,
        "fixed_seed": FIXED_SEED,
        "selected_epoch": selected_epoch,
        "model_config": model_config(profile),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "normalisation": {"mean": refit_norm.mean, "std": refit_norm.std},
        "split": {
            "selection_train": fold.train,
            "development_validation": fold.validation,
            "refit_train": refit_indices,
            "outer_test": fold.test,
        },
        "cache_sha256": file_sha256(cache),
    }
    checkpoint_path = output / "checkpoint.pt"
    atomic_checkpoint(checkpoint_path, checkpoint)
    write_history(output / "history.csv", history)
    atomic_json(output / "split.json", {
        "protocol": profile.protocol,
        "subject_one_based": fold.subject + 1,
        "outer_fold_one_based": fold.outer_fold + 1,
        "selection_train": fold.train.tolist(),
        "development_validation": fold.validation.tolist(),
        "outer_test": fold.test.tolist(),
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    replay, replay_payload = load_checkpoint(checkpoint_path, device)
    test_loader = make_loader(
        data, fold.test, refit_norm, profile,
        shuffle=False, seed=FIXED_SEED, evaluation=True,
    )
    report = predict(replay, test_loader, device, profile)
    prediction = report["probability"].argmax(axis=1).astype(np.int64)
    scores = metric_set(report["label"], prediction)
    metrics = {
        "schema": "adarcg_public_metrics_v1",
        "dataset": profile.name,
        "task": profile.task,
        "protocol": profile.protocol,
        "subject_one_based": fold.subject + 1,
        "outer_fold_one_based": fold.outer_fold + 1,
        "selected_epoch": selected_epoch,
        "parameter_count": sum(parameter.numel() for parameter in replay.parameters()),
        **scores,
    }
    atomic_predictions(
        output / "predictions.npz",
        sample_index=fold.test.astype(np.int64),
        true_label=report["label"],
        prediction=prediction,
        class_probability=report["probability"].astype(np.float32),
        subject=report["subject"],
        session=report["session"],
        trial=report["trial"],
    )
    atomic_json(output / "metrics.json", metrics)
    return metrics
