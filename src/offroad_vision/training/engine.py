"""Small, notebook-friendly PyTorch training and validation loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .metrics import confusion_matrix


def _run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_id: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_valid_pixels = 0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_pixels = int((masks != ignore_id).sum().item())

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * valid_pixels
        total_valid_pixels += valid_pixels
        matrix += confusion_matrix(
            logits.detach().cpu(),
            masks.detach().cpu(),
            num_classes=num_classes,
            ignore_id=ignore_id,
        )

    mean_loss = total_loss / max(total_valid_pixels, 1)
    return mean_loss, matrix


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    ignore_id: int = 255,
) -> tuple[float, torch.Tensor]:
    return _run_epoch(
        model, loader, criterion, device, num_classes, ignore_id, optimizer
    )


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_id: int = 255,
) -> tuple[float, torch.Tensor]:
    return _run_epoch(
        model, loader, criterion, device, num_classes, ignore_id, optimizer=None
    )
