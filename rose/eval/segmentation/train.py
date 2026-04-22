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
from rose.eval.segmentation.metrics import SEGMENTATION_METRICS
from rose.eval.segmentation.models import build_segmentation_decoder
from rose.eval.segmentation.schedulers import build_scheduler
from rose.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from rose.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


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
        pred = segmentation_model(batch_img)  # linear 头返回 tensor；M2F 头返回 dict
        if isinstance(pred, dict):
            # M2F: pred_masks [B,Q,H,W], pred_logits [B,Q,C] -> 合成 [B,C,H,W] 供 dice/ce loss
            pm = pred.get("pred_masks")
            pl = pred.get("pred_logits")
            if pm is not None and pl is not None:
                pred = torch.einsum("bqhw,bqc->bchw", pm.sigmoid(), pl.softmax(dim=-1))
            else:
                pred = pred.get("pred_masks", pred.get("pred_logits", next(iter(pred.values()))))
        gt = torch.squeeze(gt).long()  # Adapt gt dimension to enable loss calculation
        unique = torch.unique(gt)
    # c) compute loss
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    loss = criterion(pred, gt)

    # d) optimization
    max_norm = float(optimizer_gradient_clip) if isinstance(optimizer_gradient_clip, str) else optimizer_gradient_clip
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

    return loss


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
        segmentation_model.to(local_device), device_ids=[local_device]
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

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.diceloss_weight, celoss_weight=config.train.celoss_weight
    )
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    # 5- train the model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        loss = train_step(
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
        )
        global_step += 1
        metric_logger.update(loss=loss)
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
    torch.save(
        {
            "model": {k: v for k, v in segmentation_model.module.state_dict().items() if "segmentation_model.1" in k},
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values
