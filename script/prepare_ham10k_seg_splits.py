#!/usr/bin/env python3
"""
阶段 1：为 ham10k_seg 数据集生成固定的 TRAIN / VAL / TEST 划分 CSV。

ham10k_seg（rose/data/datasets/ham10k_seg.py）需要数据根目录下存在：
  HAM10000train.csv, HAM10000val.csv, HAM10000test.csv  （列名 isic_id）
  images/<isic_id>.jpg
  annotations/<isic_id>_segmentation.png   （与 isic2018._load_segmentation 一致）

用法示例：
  python script/prepare_ham10k_seg_splits.py \\
    --root /path/to/HAM10000 \\
    --seed 42

从 GroundTruth.csv 读取 ID（列名 image 或 isic_id）：
  python script/prepare_ham10k_seg_splits.py --root /path/to/HAM10000 --ground-truth GroundTruth.csv

仅扫描 images/*.jpg（无 CSV 时）：
  python script/prepare_ham10k_seg_splits.py --root /path/to/HAM10000 --from-images-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _collect_ids_from_ground_truth(root: Path, gt_name: str) -> list[str]:
    gt_path = root / gt_name if not os.path.isabs(gt_name) else Path(gt_name)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    df = pd.read_csv(gt_path)
    if "isic_id" in df.columns:
        col = "isic_id"
    elif "image" in df.columns:
        col = "image"
    else:
        raise ValueError(f"Expected column 'isic_id' or 'image' in {gt_path}, got {list(df.columns)}")
    ids = df[col].astype(str).str.strip().tolist()
    return ids


def _collect_ids_from_images(root: Path) -> list[str]:
    img_dir = root / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if not stems:
        stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No .jpg or .png under {img_dir}")
    return stems


def _check_annotations(root: Path, ids: list[str], warn_only: bool) -> None:
    ann_dir = root / "annotations"
    missing = [i for i in ids if not (ann_dir / f"{i}_segmentation.png").is_file()]
    if not missing:
        return
    msg = f"{len(missing)} ids missing {ann_dir}/<id>_segmentation.png (showing up to 5): {missing[:5]}"
    if warn_only:
        print(f"WARNING: {msg}", file=sys.stderr)
    else:
        raise FileNotFoundError(msg)


def main() -> int:
    p = argparse.ArgumentParser(description="Write HAM10000{train,val,test}.csv for ham10k_seg.")
    p.add_argument("--root", type=Path, required=True, help="HAM10000 dataset root")
    p.add_argument(
        "--ground-truth",
        type=str,
        default="GroundTruth.csv",
        help="CSV under --root with column image or isic_id (ignored if --from-images-only)",
    )
    p.add_argument(
        "--from-images-only",
        action="store_true",
        help="List IDs from images/*.jpg instead of GroundTruth.csv",
    )
    p.add_argument("--train-ratio", type=float, default=0.70, help="Train fraction")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Val fraction (test = rest)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (match SegmentationConfig.seed if desired)")
    p.add_argument(
        "--skip-annotation-check",
        action="store_true",
        help="Do not verify annotations/<id>_segmentation.png exists",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts only, do not write CSVs")
    args = p.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        return 1

    tr, vr = args.train_ratio, args.val_ratio
    if tr <= 0 or vr <= 0 or tr + vr >= 1.0:
        print("ERROR: require train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1", file=sys.stderr)
        return 1

    if args.from_images_only:
        ids = _collect_ids_from_images(root)
    else:
        try:
            ids = _collect_ids_from_ground_truth(root, args.ground_truth)
        except FileNotFoundError:
            if (root / "images").is_dir():
                print(
                    "INFO: GroundTruth not found; falling back to images/*.jpg",
                    file=sys.stderr,
                )
                ids = _collect_ids_from_images(root)
            else:
                raise

    # de-duplicate, preserve order
    seen = set()
    unique: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    ids = unique

    if not args.skip_annotation_check:
        _check_annotations(root, ids, warn_only=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ids))
    n = len(order)
    n_train = int(tr * n)
    n_val = int(vr * n)
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    if n_train + n_val >= n:
        print("ERROR: dataset too small for requested split", file=sys.stderr)
        return 1

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    def take(ix: np.ndarray) -> list[str]:
        return [ids[i] for i in ix.tolist()]

    train_ids, val_ids, test_ids = take(train_idx), take(val_idx), take(test_idx)

    print(f"Total: {n}  train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}  seed={args.seed}")

    out_train = root / "HAM10000train.csv"
    out_val = root / "HAM10000val.csv"
    out_test = root / "HAM10000test.csv"

    if args.dry_run:
        return 0

    pd.DataFrame({"isic_id": train_ids}).to_csv(out_train, index=False)
    pd.DataFrame({"isic_id": val_ids}).to_csv(out_val, index=False)
    pd.DataFrame({"isic_id": test_ids}).to_csv(out_test, index=False)
    print(f"Wrote:\n  {out_train}\n  {out_val}\n  {out_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
