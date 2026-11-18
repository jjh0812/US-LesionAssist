"""Segmentation loss and image-level evaluation metrics."""
import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersect = (probs * targets).sum(dim=dims)
    soft_dice = (2 * intersect + 1.0) / (probs.sum(dim=dims) + targets.sum(dim=dims) + 1.0)
    return bce + (1.0 - soft_dice.mean())


def scores_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.sigmoid(logits) >= 0.5
    actual = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (pred & actual).sum(dim=dims).float()
    pred_sum = pred.sum(dim=dims).float()
    actual_sum = actual.sum(dim=dims).float()
    # Both-empty masks count as perfect correct non-detection.
    dice = torch.where(pred_sum + actual_sum > 0, 2 * intersection / (pred_sum + actual_sum).clamp_min(1), 1.0)
    union = (pred | actual).sum(dim=dims).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    return dice, iou
