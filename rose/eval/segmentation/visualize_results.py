import argparse
import os
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rose.distributed as distributed
from rose.data import DatasetWithEnumeratedTargets, make_dataset
from rose.eval.segmentation.config import SegmentationConfig
from rose.eval.segmentation.inference import make_inference
from rose.eval.segmentation.models import build_segmentation_decoder
from rose.eval.segmentation.transforms import make_segmentation_eval_transforms
from rose.eval.setup import load_model_and_context


def _target_to_tensor(target):
    if isinstance(target, torch.Tensor):
        return target
    if isinstance(target, dict):
        value = target.get("data") or target.get("mask") or target.get("masks") or target.get("label")
        if value is None:
            value = next(iter(target.values()), None)
        if value is None:
            raise ValueError("segmentation target dict has no tensor value")
        return _target_to_tensor(value)
    if hasattr(target, "data") and isinstance(getattr(target, "data"), torch.Tensor):
        return getattr(target, "data")
    return torch.as_tensor(target)


def _denorm_to_uint8(img: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    img = img.detach().cpu().float()
    img = (img * std_t + mean_t).clamp(0, 255)
    return img.permute(1, 2, 0).numpy().astype(np.uint8)


def _make_overlay(img_uint8: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Red: prediction foreground, Green: GT foreground, Yellow: overlap.
    overlay = img_uint8.copy().astype(np.float32)
    pred_fg = pred > 0
    gt_fg = gt > 0
    inter = pred_fg & gt_fg
    pred_only = pred_fg & (~gt_fg)
    gt_only = gt_fg & (~pred_fg)

    overlay[pred_only] = 0.6 * overlay[pred_only] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    overlay[gt_only] = 0.6 * overlay[gt_only] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    overlay[inter] = 0.6 * overlay[inter] + 0.4 * np.array([255, 255, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_config(config_path: str) -> SegmentationConfig:
    base_cfg = OmegaConf.load(config_path)
    structured_cfg = OmegaConf.structured(SegmentationConfig)
    return OmegaConf.to_object(OmegaConf.merge(structured_cfg, base_cfg))


def main():
    parser = argparse.ArgumentParser("Visualize segmentation predictions")
    parser.add_argument("--config", required=True, help="Path to segmentation config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to decoder checkpoint (e.g. best_mIoU.pth)")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to export")
    parser.add_argument("--save-dir", required=True, help="Output directory for visualizations")
    args = parser.parse_args()

    distributed_was_enabled = distributed.is_enabled()
    if not distributed_was_enabled:
        # setup_and_build_model() expects distributed context for LR scaling rules.
        # For this standalone visualization script we run a single-process group.
        distributed.enable(overwrite=True, restrict_print_to_main_process=False)
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = _build_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        backbone, _ = load_model_and_context(cfg.model, output_dir=args.save_dir)
        segmentation_model = build_segmentation_decoder(
            backbone,
            cfg.decoder_head.backbone_out_layers,
            cfg.decoder_head.type,
            hidden_dim=cfg.decoder_head.hidden_dim,
            num_classes=cfg.decoder_head.num_classes,
            autocast_dtype=cfg.model_dtype.autocast_dtype,
            dropout=cfg.decoder_head.dropout,
        ).to(device)
        state_dict = torch.load(args.checkpoint, map_location="cpu")["model"]
        segmentation_model.load_state_dict(state_dict, strict=False)
        segmentation_model.eval()

        transforms = make_segmentation_eval_transforms(
            img_size=cfg.eval.crop_size,
            inference_mode="slide",
            use_tta=cfg.eval.use_tta,
            tta_ratios=cfg.transforms.eval.tta_ratios,
            mean=cfg.transforms.mean,
            std=cfg.transforms.std,
        )
        split_desc = cfg.datasets.val if args.split == "val" else cfg.datasets.test
        dataset = DatasetWithEnumeratedTargets(
            make_dataset(
                dataset_str=f"{split_desc}:root={cfg.datasets.root}",
                transforms=transforms,
            )
        )

        total = min(args.num_samples, len(dataset))
        indices = list(range(total))[rank::world_size]
        for i in indices:
            batch_img, (_, gt) = dataset[i]
            gt_t = _target_to_tensor(gt)
            if gt_t.ndim == 3 and gt_t.shape[0] == 1:
                gt_t = gt_t[0]
            gt_t = gt_t.long().to(device)

            # Same aggregation logic as eval.py
            aggregated_preds = torch.zeros(1, cfg.decoder_head.num_classes, gt_t.shape[-2], gt_t.shape[-1], device="cpu")
            for img in batch_img:
                img_cuda = img.unsqueeze(0).to(device=device, dtype=cfg.model_dtype.autocast_dtype)
                pred = make_inference(
                    img_cuda,
                    segmentation_model,
                    inference_mode="slide",
                    decoder_head_type=cfg.decoder_head.type,
                    rescale_to=gt_t.shape[-2:],
                    n_output_channels=cfg.decoder_head.num_classes,
                    crop_size=(cfg.eval.crop_size, cfg.eval.crop_size),
                    stride=(cfg.eval.stride, cfg.eval.stride),
                    apply_horizontal_flip=False,
                    output_activation=lambda x: torch.softmax(x, dim=1),
                ).cpu()
                aggregated_preds += pred

            pred_mask = (aggregated_preds / len(batch_img)).argmax(dim=1)[0].numpy().astype(np.uint8)
            gt_mask = gt_t.cpu().numpy().astype(np.uint8)
            img_for_vis = F.interpolate(
                batch_img[0].unsqueeze(0).float(),
                size=gt_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            img_uint8 = _denorm_to_uint8(img_for_vis, cfg.transforms.mean, cfg.transforms.std)
            overlay = _make_overlay(img_uint8, pred_mask, gt_mask)

            stem = Path(args.save_dir) / f"{i:04d}"
            Image.fromarray(img_uint8).save(f"{stem}_img.png")
            Image.fromarray((pred_mask > 0).astype(np.uint8) * 255).save(f"{stem}_pred.png")
            Image.fromarray((gt_mask > 0).astype(np.uint8) * 255).save(f"{stem}_gt.png")
            Image.fromarray(overlay).save(f"{stem}_overlay.png")

        if distributed.is_enabled():
            torch.distributed.barrier()
        if rank == 0:
            print(f"Saved {total} samples to: {args.save_dir}")
    finally:
        if not distributed_was_enabled and distributed.is_enabled():
            distributed.disable()


if __name__ == "__main__":
    main()
