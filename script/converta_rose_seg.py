#!/usr/bin/env python3
"""
One-shot converter for HAM10000/ISIC-style segmentation datasets.

Target layout (used by rose/data/datasets/ham10k_seg.py):
  <out_root>/
    images/<isic_id>.jpg
    annotations/<isic_id>_segmentation.png
    HAM10000train.csv   (column: isic_id)
    HAM10000val.csv
    HAM10000test.csv

This script is designed for "messy" Kaggle downloads where images/masks/CSV
may be in different folders or file names. It normalizes all IDs to ISIC_xxx.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ISIC_RE = re.compile(r"(ISIC_\d+)", re.IGNORECASE)
VALID_IMG_EXTS = {".jpg", ".jpeg", ".png"}
VALID_MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_isic_id(text: str) -> str | None:
    m = ISIC_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _collect_images(images_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in _iter_files(images_root):
        if p.suffix.lower() not in VALID_IMG_EXTS:
            continue
        isic_id = _normalize_isic_id(p.stem)
        if not isic_id:
            continue
        # Prefer first occurrence, warn for duplicates.
        if isic_id in mapping:
            print(f"[WARN] duplicate image id {isic_id}: keep {mapping[isic_id]}, skip {p}", file=sys.stderr)
            continue
        mapping[isic_id] = p
    return mapping


def _collect_masks(masks_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in _iter_files(masks_root):
        if p.suffix.lower() not in VALID_MASK_EXTS:
            continue
        # Accept file names like:
        #   ISIC_0000001_segmentation.png
        #   ISIC_0000001.png
        #   xxx_ISIC_0000001_mask.png
        base = p.stem
        isic_id = _normalize_isic_id(base)
        if not isic_id:
            continue
        if isic_id in mapping:
            print(f"[WARN] duplicate mask id {isic_id}: keep {mapping[isic_id]}, skip {p}", file=sys.stderr)
            continue
        mapping[isic_id] = p
    return mapping


def _read_ids_from_csv(csv_path: Path, column_hint: str | None) -> list[str]:
    df = pd.read_csv(csv_path)
    candidates = []
    if column_hint and column_hint in df.columns:
        candidates.append(column_hint)
    candidates.extend([c for c in ("isic_id", "image", "image_id") if c in df.columns and c not in candidates])
    if not candidates:
        raise ValueError(
            f"Cannot find ID column in {csv_path}. "
            f"Please pass --id-column. Available columns: {list(df.columns)}"
        )
    col = candidates[0]
    ids = []
    for raw in df[col].astype(str).tolist():
        isic_id = _normalize_isic_id(raw) or raw.strip()
        if isic_id:
            ids.append(isic_id.upper())
    return ids


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _write_split_csv(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["isic_id"])
        writer.writeheader()
        for isic_id in ids:
            writer.writerow({"isic_id": isic_id})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle-like HAM10000 data to ROSE ham10k_seg format."
    )
    parser.add_argument("--images-root", type=Path, required=True, help="Root directory containing image files.")
    parser.add_argument("--masks-root", type=Path, required=True, help="Root directory containing mask files.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place files into out-root/images and annotations.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV used as source ID list before train/val/test split.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional ID column name in --split-csv (e.g. isic_id/image/image_id).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any ID in split source lacks image or mask. Default: skip missing IDs with warnings.",
    )
    args = parser.parse_args()

    if not args.images_root.is_dir():
        print(f"[ERROR] --images-root is not a directory: {args.images_root}", file=sys.stderr)
        return 1
    if not args.masks_root.is_dir():
        print(f"[ERROR] --masks-root is not a directory: {args.masks_root}", file=sys.stderr)
        return 1
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1.0:
        print("[ERROR] Need train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.", file=sys.stderr)
        return 1

    print("[INFO] scanning images...")
    image_map = _collect_images(args.images_root)
    print(f"[INFO] found image IDs: {len(image_map)}")

    print("[INFO] scanning masks...")
    mask_map = _collect_masks(args.masks_root)
    print(f"[INFO] found mask IDs: {len(mask_map)}")

    if args.split_csv:
        if not args.split_csv.is_file():
            print(f"[ERROR] --split-csv not found: {args.split_csv}", file=sys.stderr)
            return 1
        base_ids = _read_ids_from_csv(args.split_csv, args.id_column)
        # de-duplicate while preserving order
        seen = set()
        ids = []
        for i in base_ids:
            if i not in seen:
                seen.add(i)
                ids.append(i)
        print(f"[INFO] IDs from split CSV: {len(ids)}")
    else:
        ids = sorted(set(image_map).intersection(mask_map))
        print(f"[INFO] IDs from image-mask intersection: {len(ids)}")

    usable_ids = []
    missing = 0
    for isic_id in ids:
        has_img = isic_id in image_map
        has_mask = isic_id in mask_map
        if has_img and has_mask:
            usable_ids.append(isic_id)
        else:
            missing += 1
            msg = f"[WARN] missing pair for {isic_id}: image={has_img}, mask={has_mask}"
            if args.strict:
                print(f"[ERROR] {msg}", file=sys.stderr)
                return 1
            print(msg, file=sys.stderr)

    if len(usable_ids) < 3:
        print(f"[ERROR] too few usable IDs: {len(usable_ids)}", file=sys.stderr)
        return 1

    out_root = args.out_root.resolve()
    out_images = out_root / "images"
    out_masks = out_root / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] materializing files using mode={args.mode}...")
    for i, isic_id in enumerate(usable_ids, 1):
        img_src = image_map[isic_id]
        mask_src = mask_map[isic_id]
        img_dst = out_images / f"{isic_id}.jpg"
        mask_dst = out_masks / f"{isic_id}_segmentation.png"
        _link_or_copy(img_src.resolve(), img_dst, args.mode)
        _link_or_copy(mask_src.resolve(), mask_dst, args.mode)
        if i % 1000 == 0:
            print(f"[INFO] processed {i}/{len(usable_ids)}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(usable_ids))
    n = len(order)
    n_train = max(int(args.train_ratio * n), 1)
    n_val = max(int(args.val_ratio * n), 1)
    if n_train + n_val >= n:
        print("[ERROR] dataset too small for requested split ratios.", file=sys.stderr)
        return 1

    train_ids = [usable_ids[i] for i in order[:n_train].tolist()]
    val_ids = [usable_ids[i] for i in order[n_train : n_train + n_val].tolist()]
    test_ids = [usable_ids[i] for i in order[n_train + n_val :].tolist()]

    _write_split_csv(out_root / "HAM10000train.csv", train_ids)
    _write_split_csv(out_root / "HAM10000val.csv", val_ids)
    _write_split_csv(out_root / "HAM10000test.csv", test_ids)

    print("[DONE] conversion finished.")
    print(f"  out_root: {out_root}")
    print(f"  usable IDs: {len(usable_ids)} (missing skipped: {missing})")
    print(f"  split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
