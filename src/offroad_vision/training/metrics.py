"""Pixel confusion matrices and IoU summaries for segmentation models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_id: int = 255,
) -> torch.Tensor:
    """Compute rows=ground truth, columns=prediction pixel counts."""
    if predictions.ndim == targets.ndim + 1:
        predictions = predictions.argmax(dim=1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {predictions.shape}, {targets.shape}"
        )

    predictions = predictions.reshape(-1).to(torch.int64)
    targets = targets.reshape(-1).to(torch.int64)
    valid = (
        (targets != ignore_id)
        & (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    indices = targets[valid] * num_classes + predictions[valid]
    return torch.bincount(indices, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def class_iou(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(torch.float64)
    intersection = matrix.diag()
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - intersection
    return torch.where(
        union > 0,
        intersection / union,
        torch.full_like(union, torch.nan),
    )


def _nanmean(values: torch.Tensor) -> float:
    valid = ~torch.isnan(values)
    return float(values[valid].mean()) if valid.any() else float("nan")


def segmentation_summary(
    matrix: torch.Tensor,
    class_names: Mapping[int, str],
    navigation_ids: Iterable[int],
) -> dict[str, object]:
    iou = class_iou(matrix)
    navigation_ids = tuple(navigation_ids)
    total = matrix.sum()
    accuracy = (
        float(matrix.diag().sum() / total) if int(total.item()) > 0 else float("nan")
    )

    return {
        "pixel_accuracy": accuracy,
        "all_class_miou": _nanmean(iou),
        "navigation_miou": _nanmean(iou[list(navigation_ids)]),
        "per_class_iou": {
            class_names[class_id]: float(iou[class_id])
            for class_id in sorted(class_names)
        },
    }
