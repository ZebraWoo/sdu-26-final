import json
import logging
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from omegaconf import MISSING
PROJECT_ROOT = "/home/jiangwenjing/hd/rose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rose.data import SamplerType, make_data_loader, make_dataset
from rose.data.adapters import DatasetWithEnumeratedTargets
from rose.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)
from rose.configs import DinoV3SetupArgs, config, setup_config
from rose.models import build_model_for_eval
from rose.eval.helpers import cli_parser
from rose.eval.metrics import ClassificationMetricType, build_classification_metric
from rose.eval.setup import ModelConfig, load_model_and_context
from rose.eval.utils import LossType, ModelWithIntermediateLayers, average_metrics, evaluate
from rose.models import panderm


def benchmark_launcher(eval_args: dict[str, object]):
    #dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=LinearEvalConfig)
    setup_args = DinoV3SetupArgs(
        config_file=eval_args.get("config_file"),
        pretrained_weights=eval_args.get("pretrained_weights", None),
        shard_unsharded_model=False,
        output_dir=None,
        opts=[],
    )
    config = setup_config(setup_args, strict_cfg=False, dist=False)
    if eval_args["model"] == "rose":
        model = build_model_for_eval(config, setup_args.pretrained_weights)
    elif eval_args["model"] == "panderm":
        model = panderm.panderm_large_patch16_224(pretrained=True)
        state_dict = torch.load(eval_args.get("pretrained_weights"), map_location='cpu', weights_only=True)
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=torch.float32)
    feature_model = ModelWithIntermediateLayers(feature_model=model, n_last_blocks=1, autocast_ctx=autocast_ctx, reshape=False, return_class_token=True)
    test_loader = build_loader(eval_args)
    all_features, all_labels = extract_features(model=feature_model, test_loader=test_loader)
    rng = np.random.default_rng(seed=42)

    labels = all_labels
    features = all_features

    selected_indices = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]

        if len(cls_idx) > 200:
            cls_idx = rng.choice(cls_idx, size=200, replace=False)

        selected_indices.append(cls_idx)

    # 合并所有类别的索引
    selected_indices = np.concatenate(selected_indices)

    # 应用采样
    all_features = features[selected_indices]
    all_labels   = labels[selected_indices]

    print("After per-class sampling:")
    print(all_features.shape, all_labels.shape)
    # -------- t-SNE --------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=42,
    )

    features_2d = tsne.fit_transform(all_features)  # [N, 2]

    # -------- 绘图 --------
    plt.figure(figsize=(6, 4.5))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=all_labels,
        marker="o",
        cmap="tab10",
        s=13,
        alpha=0.8,
    )

    plt.colorbar(scatter)
    # plt.axis("off")
    #plt.xlabel("t-SNE Dim 1")
    #plt.ylabel("t-SNE Dim 2")
    #plt.title("t-SNE of CLS + Mean Patch Features")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("/home/jiangwenjing/hd/rose/article_issue/article_figures/rose_hiba_tsne.pdf", dpi=300)
    #plt.show()
    return 0

def extract_features(model, test_loader):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            features = model(data)
            # -------- 取 ViT 特征 --------
            patch_tokens = features[0][0]   # [B, 196, 1024]
            cls_token    = features[0][1]   # [B, 1024]

            # patch 平均池化
            #patch_feat = patch_tokens.mean(dim=1)  # [B, 1024]

            # CLS + patch concat
            #global_feat = torch.cat([cls_token, patch_feat], dim=1)  # [B, 2048]

            all_features.append(cls_token.cpu())
            all_labels.append(labels.cpu())
    # -------- 拼接所有 batch --------  
    all_features = torch.cat(all_features, dim=0).numpy()  # [N, 2048]
    all_labels = torch.cat(all_labels, dim=0).numpy()      # [N]
    print(all_features.shape)
    return all_features, all_labels

def build_loader(args):
    transform = make_classification_eval_transform(resize_size=256, crop_size=224)
    test_dataset = make_dataset(dataset_str=args["dataset"], transform=transform)
    test_dataloader = make_data_loader(
        dataset=test_dataset,
        batch_size= 128,
        shuffle=False,
        num_workers=8,
        sampler_type=None,
        drop_last=False,
    )

    return test_dataloader
    

def main(argv=None):
    if argv is None:
        argv = [
            "model=rose",
            "config_file=/data/wenjing/skin_dataset/rose-outputs/config.yaml",
            "pretrained_weights=/data/wenjing/skin_dataset/rose-outputs/eval/training_35999/teacher_checkpoint.pth",
            "dataset=HIBA:split=TRAIN:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra",
        ]
        # argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    benchmark_launcher(eval_args=eval_args)
    return 0

if __name__ == "__main__":
    main()