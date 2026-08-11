"""Channel order and the anatomical region prior used by AdaRCG."""

from __future__ import annotations

import numpy as np


CHANNELS = (
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
    "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2",
)

REGIONS = (
    "frontal_left",
    "frontal_right",
    "temporal_left",
    "temporal_right",
    "central",
    "parietal",
    "occipital",
)


def _side(name: str) -> str:
    if name.endswith("z"):
        return "mid"
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        return "mid"
    return "left" if int(digits) % 2 else "right"


def anatomical_membership() -> np.ndarray:
    membership = np.zeros((len(REGIONS), len(CHANNELS)), dtype=np.float32)
    for channel, name in enumerate(CHANNELS):
        side = _side(name)
        if name.startswith(("Fp", "AF")) or (
            name.startswith("F") and not name.startswith(("FC", "FT"))
        ):
            if side == "left":
                membership[0, channel] = 1.0
            elif side == "right":
                membership[1, channel] = 1.0
            else:
                membership[0:2, channel] = 0.5
        elif name.startswith(("FT", "T", "TP")):
            membership[2 if side == "left" else 3, channel] = 1.0
        elif name.startswith(("FC", "C", "CP")):
            membership[4, channel] = 1.0
        elif name.startswith(("P", "PO")):
            membership[5, channel] = 1.0
        elif name.startswith(("O", "CB")):
            membership[6, channel] = 1.0
        else:
            raise ValueError(f"unmapped EEG channel: {name}")
    if not np.allclose(membership.sum(axis=0), 1.0):
        raise AssertionError("each channel must have unit region membership")
    return membership


def region_graph() -> np.ndarray:
    edges = (
        (0, 1), (0, 2), (1, 3), (0, 4), (1, 4), (2, 4),
        (3, 4), (2, 5), (3, 5), (4, 5), (5, 6),
    )
    adjacency = np.eye(len(REGIONS), dtype=np.float32)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    return adjacency


def dataset_regions(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset.lower().replace("-", "") not in {"seed", "mped", "seediv", "seedv"}:
        raise ValueError(f"unsupported dataset: {dataset}")
    return anatomical_membership(), region_graph()

