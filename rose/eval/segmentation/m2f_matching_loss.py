from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover - surfaced with a clear runtime error
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def _pairwise_sigmoid_ce_cost(out_prob: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    # out_prob: [Q, HW], tgt_mask: [N, HW]
    out_prob = out_prob.float()
    tgt_mask = tgt_mask.float()
    eps = 1e-6
    out_prob = out_prob.clamp(min=eps, max=1 - eps)
    pos = -torch.matmul(torch.log(out_prob), tgt_mask.t())
    neg = -torch.matmul(torch.log(1 - out_prob), (1 - tgt_mask).t())
    return (pos + neg) / out_prob.shape[1]


def _pairwise_dice_cost(out_prob: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    # out_prob: [Q, HW], tgt_mask: [N, HW]
    out_prob = out_prob.float()
    tgt_mask = tgt_mask.float()
    num = 2 * torch.matmul(out_prob, tgt_mask.t())
    den = out_prob.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    return 1 - (num + 1.0) / (den + 1.0)


@dataclass
class M2FLossWeights:
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    eos_coef: float = 0.1


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight: float = 2.0, mask_weight: float = 5.0, dice_weight: float = 5.0):
        super().__init__()
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if linear_sum_assignment is None:
            raise RuntimeError(f"scipy is required for Hungarian matching: {_SCIPY_IMPORT_ERROR}")

        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
        bs, num_queries = pred_logits.shape[:2]
        out_prob_cls = pred_logits.softmax(dim=-1)

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(bs):
            tgt_labels = targets[b]["labels"]  # [N]
            tgt_masks = targets[b]["masks"]  # [N, Ht, Wt]
            n_tgt = tgt_labels.numel()
            if n_tgt == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                        torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                    )
                )
                continue

            out_prob = pred_masks[b].sigmoid()
            if out_prob.shape[-2:] != tgt_masks.shape[-2:]:
                out_prob = F.interpolate(
                    out_prob.unsqueeze(0),
                    size=tgt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]

            out_flat = out_prob.flatten(1).float()  # [Q, HW]
            tgt_flat = tgt_masks.float().flatten(1)  # [N, HW]

            cost_class = -out_prob_cls[b][:, tgt_labels].float()  # [Q, N]
            cost_mask = _pairwise_sigmoid_ce_cost(out_flat, tgt_flat)  # [Q, N]
            cost_dice = _pairwise_dice_cost(out_flat, tgt_flat)  # [Q, N]

            total_cost = (
                self.class_weight * cost_class
                + self.mask_weight * cost_mask
                + self.dice_weight * cost_dice
            )
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=pred_logits.device),
                )
            )
        return indices


class M2FSetCriterion(nn.Module):
    def __init__(self, num_classes: int, weights: M2FLossWeights):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            class_weight=weights.class_weight,
            mask_weight=weights.mask_weight,
            dice_weight=weights.dice_weight,
        )
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = weights.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.weights = weights

    @staticmethod
    def _build_targets(gt: torch.Tensor) -> List[dict]:
        # gt: [B, H, W] with classes {0,1} for HAM10K seg.
        targets: List[dict] = []
        for b in range(gt.shape[0]):
            fg = (gt[b] > 0)
            if fg.any():
                targets.append(
                    {
                        "labels": torch.tensor([1], dtype=torch.long, device=gt.device),
                        "masks": fg.unsqueeze(0).float(),
                    }
                )
            else:
                targets.append(
                    {
                        "labels": torch.empty(0, dtype=torch.long, device=gt.device),
                        "masks": torch.empty(0, gt.shape[-2], gt.shape[-1], dtype=torch.float32, device=gt.device),
                    }
                )
        return targets

    def _loss_labels(self, outputs: dict, targets: List[dict], indices):
        src_logits = outputs["pred_logits"]  # [B,Q,C+1]
        bs, nq = src_logits.shape[:2]
        target_classes = torch.full(
            (bs, nq),
            self.num_classes,  # no-object index
            dtype=torch.int64,
            device=src_logits.device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(device=src_logits.device, dtype=src_logits.dtype),
        )
        return loss_ce

    def _loss_masks(self, outputs: dict, targets: List[dict], indices):
        src_masks = outputs["pred_masks"]  # [B,Q,H,W]
        matched_src = []
        matched_tgt = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            matched_src.append(src_masks[b, src_idx])
            matched_tgt.append(targets[b]["masks"][tgt_idx])
        if not matched_src:
            zero = src_masks.sum() * 0.0
            return zero, zero

        src = torch.cat(matched_src, dim=0)  # [M,H,W]
        tgt = torch.cat(matched_tgt, dim=0)  # [M,H,W]
        if src.shape[-2:] != tgt.shape[-2:]:
            src = F.interpolate(src.unsqueeze(1), size=tgt.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        loss_mask = F.binary_cross_entropy_with_logits(src, tgt, reduction="mean")
        src_prob = src.sigmoid().flatten(1)
        tgt_flat = tgt.flatten(1)
        numerator = 2 * (src_prob * tgt_flat).sum(-1)
        denominator = src_prob.sum(-1) + tgt_flat.sum(-1)
        loss_dice = 1 - (numerator + 1.0) / (denominator + 1.0)
        return loss_mask, loss_dice.mean()

    def _compute_losses(self, outputs: dict, targets: List[dict], return_components: bool = False):
        indices = self.matcher(outputs, targets)
        loss_ce = self._loss_labels(outputs, targets, indices)
        loss_mask, loss_dice = self._loss_masks(outputs, targets, indices)
        total = (
            self.weights.class_weight * loss_ce
            + self.weights.mask_weight * loss_mask
            + self.weights.dice_weight * loss_dice
        )
        if not return_components:
            return total
        components = {
            "loss_cls": loss_ce,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_cls_weighted": self.weights.class_weight * loss_ce,
            "loss_mask_weighted": self.weights.mask_weight * loss_mask,
            "loss_dice_weighted": self.weights.dice_weight * loss_dice,
            "loss_total_matching": total,
        }
        return total, components

    def forward(self, outputs: dict, gt: torch.Tensor, return_components: bool = False):
        targets = self._build_targets(gt)
        if return_components:
            total, components = self._compute_losses(outputs, targets, return_components=True)
        else:
            total = self._compute_losses(outputs, targets, return_components=False)
            components = None
        for aux in outputs.get("aux_outputs", []):
            if return_components:
                aux_total, aux_components = self._compute_losses(aux, targets, return_components=True)
                total = total + aux_total
                for key in ("loss_cls", "loss_mask", "loss_dice", "loss_cls_weighted", "loss_mask_weighted", "loss_dice_weighted", "loss_total_matching"):
                    components[key] = components[key] + aux_components[key]
            else:
                total = total + self._compute_losses(aux, targets, return_components=False)
        if return_components:
            components["loss_total_with_aux"] = total
            return total, components
        return total
