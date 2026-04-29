# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import numpy as np
import os
import random

import torch
import torch.distributed as dist

from rose.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import rose.distributed as distributed
from rose.eval.segmentation.eval import evaluate_segmentation_model
from rose.eval.segmentation.loss import MultiSegmentationLoss
from rose.eval.segmentation.m2f_matching_loss import M2FSetCriterion, M2FLossWeights
from rose.eval.segmentation.metrics import SEGMENTATION_METRICS
from rose.eval.segmentation.models import build_segmentation_decoder
from rose.eval.segmentation.schedulers import build_scheduler
from rose.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from rose.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


def _tensor_stats(x: torch.Tensor) -> str:
    x = x.detach()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(x.min()):.4f} max={float(x.max()):.4f} mean={float(x.mean()):.4f}"
    )


def _log_dataflow(
    *,
    global_step: int,
    batch_img: torch.Tensor,
    gt: torch.Tensor,
    raw_pred,
    pred_for_loss: torch.Tensor,
    monitor_cfg,
    loss_components: dict | None = None,
):
    if not getattr(monitor_cfg, "enabled", False):
        return
    interval = max(int(getattr(monitor_cfg, "interval", 200)), 1)
    if global_step % interval != 0:
        return
    if getattr(monitor_cfg, "only_rank0", True) and not distributed.is_main_process():
        return

    msg = [f"[MONITOR] step={global_step}"]
    gt_fg_ratio = float((gt > 0).float().mean())
    msg.append(f"gt_fg_ratio={gt_fg_ratio:.4f}")

    if isinstance(raw_pred, dict):
        pm = raw_pred.get("pred_masks")
        pl = raw_pred.get("pred_logits")
        if isinstance(pl, torch.Tensor):
            probs = pl.softmax(dim=-1).detach()
            noobj_prob = float(probs[..., -1].mean())
            fg_prob = float(probs[..., :-1].mean())
            msg.append(f"class_prob_mean(fg={fg_prob:.4f},noobj={noobj_prob:.4f})")
            if getattr(monitor_cfg, "log_pred_logits_softmax", False):
                q = max(int(getattr(monitor_cfg, "pred_logits_softmax_queries", 4)), 1)
                q = min(q, probs.shape[1])
                probs_sample = probs[0, :q, :].float().cpu()
                probs_sample = torch.round(probs_sample * 10000) / 10000
                msg.append(f"softmax[0,:{q},:]={probs_sample.tolist()}")

    pred_fg_ratio = float((pred_for_loss.argmax(dim=1) > 0).float().mean())
    msg.append(f"pred_fg_ratio={pred_fg_ratio:.4f}")
    if loss_components:
        printed = []
        for key in (
            "loss_cls",
            "loss_mask",
            "loss_dice",
            "loss_cls_weighted",
            "loss_mask_weighted",
            "loss_dice_weighted",
            "loss_total_matching",
            "loss_semantic_ce_aux",
        ):
            if key in loss_components:
                value = loss_components[key]
                if isinstance(value, torch.Tensor):
                    value = float(value.detach())
                else:
                    value = float(value)
                printed.append(f"{key}={value:.4f}")
        if printed:
            msg.append("losses{" + ", ".join(printed) + "}")
    logger.info(" | ".join(msg))


def _seg_target_to_tensor(t):
    """从 Mask / dict / tensor 中取出普通 tensor，供 collate 与 train_step 使用。"""
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, dict):
        v = t.get("data") or t.get("mask") or t.get("masks") or t.get("label")
        if v is not None:
            return _seg_target_to_tensor(v)
        v = next(iter(t.values()), None)
        if v is not None:
            return _seg_target_to_tensor(v)
        raise ValueError("seg target dict has no tensor inside")
    # tv_tensors.Mask 等可能是 Tensor 子类，有 .data 等
    if hasattr(t, "data") and isinstance(getattr(t, "data"), torch.Tensor):
        return getattr(t, "data")
    return torch.as_tensor(t)


def seg_collate_fn(batch):
    """Collate (image, (index, target)) 为 (images, (indices, targets))，target 转为普通 tensor。"""
    images = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1][0] for b in batch], dtype=torch.long)
    targets = torch.stack([_seg_target_to_tensor(b[1][1]) for b in batch])
    return images, (indices, targets)


class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _get_backbone_and_head_parameters(segmentation_model: torch.nn.Module):
    backbone_params = [p for p in segmentation_model.module.segmentation_model[0].parameters() if p.requires_grad]
    head_params = [p for p in segmentation_model.module.segmentation_model[1].parameters() if p.requires_grad]
    return backbone_params, head_params


def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
    mode,
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
        mode,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    # `segmentation_model` is a module list of [backbone, decoder]
    # Only put the head in train mode
    segmentation_model.module.segmentation_model[1].train()
    is_better = False
    if new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value:
        is_better = True
    return is_better, new_metric_values_dict


def train_step(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion,
    model_dtype,
    global_step,
    monitor_cfg=None,
    decoder_head_type: str = "linear",
    semantic_ce_aux_weight: float = 0.0,
    freeze_backbone: bool = False,
    backbone_group_idx: int = 0,
):
    # a) load batch
    batch_img, (_, gt) = batch

    # tv_tensors.Mask 等经 default_collate 后可能变成 dict，需取出张量
    while isinstance(gt, dict):
        gt = gt.get("data") or gt.get("mask") or gt.get("masks") or gt.get("label") or next(iter(gt.values()), None)
        if gt is None:
            raise ValueError("batch target is dict but no tensor found inside")
    batch_img = batch_img.to(device)  # B x C x h x w
    gt = gt.to(device)  # B x (num_classes if multilabel) x h x w
    optimizer.zero_grad(set_to_none=True)

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        raw_pred = segmentation_model(batch_img)  # linear 头返回 tensor；M2F 头返回 dict
        pred = raw_pred
        if isinstance(raw_pred, dict):
            # M2F: pred_masks [B,Q,H,W], pred_logits [B,Q,C] -> 合成 [B,C,H,W] 供 dice/ce loss
            pm = raw_pred.get("pred_masks")
            pl = raw_pred.get("pred_logits")
            if pm is not None and pl is not None:
                # Keep consistent with inference: drop the "no-object" class at last logit channel.
                mask_cls = pl.softmax(dim=-1)[..., :-1]
                mask_pred = pm.sigmoid()
                pred = torch.einsum("bqc,bqhw->bchw", mask_cls.to(torch.float), mask_pred.to(torch.float))
            else:
                pred = pred.get("pred_masks", pred.get("pred_logits", next(iter(pred.values()))))
        # Keep batch dimension for bs=1; only squeeze a singleton channel if present.
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0, ...]
        elif gt.ndim == 2:
            gt = gt.unsqueeze(0)
        gt = gt.long()
        unique = torch.unique(gt)
    # c) compute loss
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    loss_components = None
    if decoder_head_type == "m2f" and isinstance(raw_pred, dict) and isinstance(criterion, M2FSetCriterion):
        do_monitor_log = bool(getattr(monitor_cfg, "enabled", False)) and (
            global_step % max(int(getattr(monitor_cfg, "interval", 200)), 1) == 0
        )
        if do_monitor_log:
            loss, loss_components = criterion(raw_pred, gt, return_components=True)
        else:
            loss = criterion(raw_pred, gt)
        if semantic_ce_aux_weight > 0:
            # Use synthesized [B,C,H,W] output as logits directly to avoid
            # bf16 logit overflow/NaN when values are outside [0, 1].
            semantic_logits = pred.float()
            num_classes = semantic_logits.shape[1]
            semantic_target = gt.clone()
            invalid = (semantic_target < 0) | (semantic_target >= num_classes)
            if invalid.any():
                # Some datasets may contain ignore/unlabeled ids (e.g. 255).
                # Mark out-of-range labels as ignore_index for CE.
                semantic_target[invalid] = 255
            semantic_ce = torch.nn.functional.cross_entropy(
                semantic_logits,
                semantic_target,
                ignore_index=255,
            )
            loss = loss + semantic_ce_aux_weight * semantic_ce
            if loss_components is not None:
                loss_components["loss_semantic_ce_aux"] = semantic_ce_aux_weight * semantic_ce
    else:
        loss = criterion(pred, gt)
    _log_dataflow(
        global_step=global_step,
        batch_img=batch_img,
        gt=gt,
        raw_pred=raw_pred,
        pred_for_loss=pred,
        monitor_cfg=monitor_cfg,
        loss_components=loss_components,
    )
    # Track GT foreground ratio to quickly detect label-collapse (all background).
    fg_ratio = (gt > 0).float().mean()

    # d) optimization
    max_norm = float(optimizer_gradient_clip) if isinstance(optimizer_gradient_clip, str) else optimizer_gradient_clip
    if freeze_backbone and 0 <= backbone_group_idx < len(optimizer.param_groups):
        # Keep forward/backward graph unchanged for DDP+checkpoint compatibility,
        # but freeze backbone update by setting its optimizer LR to zero.
        optimizer.param_groups[backbone_group_idx]["lr"] = 0.0
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), max_norm)
        optimizer.step()

    if global_step > 0:  # inheritance from old mmcv code
        scheduler.step()

    return loss, fg_ratio


def train_segmentation(
    backbone,
    config,
):
    # 支持 linear 和 m2f 两种 decoder 头类型
    if config.decoder_head.type not in ["linear", "m2f"]:
        raise ValueError(f"Unsupported decoder_head.type={config.decoder_head.type}")

    # 1- load the segmentation decoder
    logger.info("Initializing the segmentation model")
    decoder_type = config.decoder_head.type

    # 对于 m2f 需要 hidden_dim，linear 使用 dropout；多传参数通常也会被忽略，这里区分更清晰
    decoder_kwargs = dict(
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=config.model_dtype.autocast_dtype,
    )
    if decoder_type == "linear":
        decoder_kwargs["dropout"] = config.decoder_head.dropout
    elif decoder_type == "m2f":
        decoder_kwargs["hidden_dim"] = config.decoder_head.hidden_dim

    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        decoder_type,
        **decoder_kwargs,
    )
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device),
        device_ids=[local_device],
    )  # should be local rank
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")

    # 2- create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
        reduce_zero_label=config.eval.reduce_zero_label,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}:root={config.datasets.root}",
            transforms=train_transforms,
        )
    )
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
            collate_fn=seg_collate_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=val_transforms,
        )
    )
    val_sampler_type = None
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- define and create scaler, optimizer, scheduler, loss
    scaler = None
    if config.model_dtype.autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    backbone_lr = config.optimizer.lr * config.train.backbone_lr_multiplier
    backbone_params, head_params = _get_backbone_and_head_parameters(segmentation_model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": head_params,
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            },
        ]
    )
    logger.info(
        "Optimizer param groups: "
        f"backbone_lr={backbone_lr:.3e}, head_lr={config.optimizer.lr:.3e}, "
        f"backbone_params={sum(p.numel() for p in backbone_params)}, "
        f"head_params={sum(p.numel() for p in head_params)}"
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    if decoder_type == "m2f" and getattr(config.train, "use_m2f_matching_loss", False):
        criterion = M2FSetCriterion(
            num_classes=config.decoder_head.num_classes,
            weights=M2FLossWeights(
                class_weight=config.train.m2f_cls_weight,
                mask_weight=config.train.m2f_mask_weight,
                dice_weight=config.train.m2f_dice_weight,
                eos_coef=config.train.m2f_eos_coef,
            ),
        )
        logger.info(
            "Using M2F matching loss (Hungarian): "
            f"cls={config.train.m2f_cls_weight}, mask={config.train.m2f_mask_weight}, "
            f"dice={config.train.m2f_dice_weight}, eos={config.train.m2f_eos_coef}"
        )
    else:
        criterion = MultiSegmentationLoss(
            diceloss_weight=config.train.diceloss_weight, celoss_weight=config.train.celoss_weight
        )
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}
    no_improve_evals = 0
    early_stop_patience = config.eval.early_stop_patience

    def _save_decoder_checkpoint(filename: str, metric_values: dict | None = None):
        torch.save(
            {
                "model": {k: v for k, v in segmentation_model.module.state_dict().items() if "segmentation_model.1" in k},
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "metrics": metric_values or {},
            },
            os.path.join(config.output_dir, filename),
        )

    # 5- train the model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    metric_logger.add_meter("fg_ratio", SmoothedValue(window_size=20, fmt="{value:.4f}"))
    freeze_backbone_iters = max(int(getattr(config.train, "freeze_backbone_iters", 0)), 0)
    if freeze_backbone_iters > 0:
        logger.info(f"Backbone frozen for first {freeze_backbone_iters} iterations (by lr=0)")
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        if freeze_backbone_iters > 0 and global_step == freeze_backbone_iters:
            logger.info(f"Backbone unfrozen at iteration {global_step}")
        is_backbone_frozen = freeze_backbone_iters > 0 and global_step < freeze_backbone_iters
        loss, fg_ratio = train_step(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            config.model_dtype.autocast_dtype,
            global_step,
            config.monitor,
            decoder_type,
            config.train.semantic_ce_aux_weight,
            is_backbone_frozen,
            0,
        )
        global_step += 1
        metric_logger.update(loss=loss, fg_ratio=float(fg_ratio))
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                mode="train",
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict
                no_improve_evals = 0
                _save_decoder_checkpoint("best_mIoU.pth", metric_values=best_metric_values_dict)
            else:
                no_improve_evals += 1
                if early_stop_patience is not None and no_improve_evals >= early_stop_patience:
                    logger.info(
                        f"Early stopping triggered at step {global_step}: "
                        f"no improvement for {no_improve_evals} eval rounds."
                    )
                    break

        # one last validation only if the number of total iterations is NOT divisible by eval interval:
        if total_iter % config.eval.eval_interval:
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                config.model_dtype.autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
                mode="train",
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict
    logger.info("Training is done!")
    # segmentation_model is a module list of [backbone, decoder]
    # Only save the decoder head
    _save_decoder_checkpoint("model_final.pth", metric_values=global_best_metric_values)
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values
