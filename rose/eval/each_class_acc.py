import os
import glob
import numpy as np
import torch

ROOT_DIR = "/data/wenjing/skin_dataset/swavderm_eval"

# ------------------------------------------------
# Compute per-class accuracy
# ------------------------------------------------
def compute_per_class_accuracy(preds, targets):
    """
    Args:
        preds: numpy array, shape (N, C) or (N,)
        targets: numpy array, shape (N,)
    Returns:
        dict: {class_id: accuracy}
    """
    if preds.ndim == 2:
        preds_label = np.argmax(preds, axis=1)
    else:
        preds_label = preds

    num_classes = int(targets.max()) + 1
    acc_dict = {}

    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            acc_dict[c] = np.nan
        else:
            acc_dict[c] = (preds_label[mask] == targets[mask]).mean()

    return acc_dict


# ------------------------------------------------
# Scan all *_linear_output folders
# ------------------------------------------------
linear_dirs = sorted(
    d for d in glob.glob(os.path.join(ROOT_DIR, "*_linear_output"))
    if os.path.isdir(d)
)

print(f"Found {len(linear_dirs)} linear output folders.\n")

# ------------------------------------------------
# Loop over each folder
# ------------------------------------------------
for linear_dir in linear_dirs:
    print("=" * 72)
    print(f"Processing: {linear_dir}")

    preds_files = glob.glob(os.path.join(linear_dir, "preds*TEST.npy"))
    target_files = glob.glob(os.path.join(linear_dir, "target*TEST.npy"))

    if len(preds_files) == 0 or len(target_files) == 0:
        print("  ⚠️ Missing preds or target file, skipped.")
        continue

    preds_path = preds_files[0]
    target_path = target_files[0]

    preds = np.load(preds_path)
    targets = np.load(target_path)

    print(f"  Loaded preds:   {preds.shape}")
    print(f"  Loaded targets: {targets.shape}")

    # If logits → softmax
    if preds.ndim == 2 and (preds.max() > 1.0 or preds.min() < 0):
        preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()

    acc_dict = compute_per_class_accuracy(preds, targets)

    print("  Per-class accuracy:")
    for c, acc in acc_dict.items():
        if np.isnan(acc):
            print(f"    Class {c}: N/A")
        else:
            print(f"    Class {c}: {acc:.4f}")

print("\nDone.")
