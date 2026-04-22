import numpy as np
import torch
import csv
import os
from pathlib import Path
import argparse
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
)
from torchmetrics.functional.classification.auroc import _reduce_auroc
import re
from pathlib import Path
# -------------------------
# 1. Extend MulticlassPrecisionRecallCurve to compute AUPR
# -------------------------
def compute_AUPR(self, num_classes=None, curve=None):
    """
    Return AUPR , without plotting.
    If num_classes==2, then only report score[1], because PR curve is not symmetrical like ROC.
    """
    if num_classes == 2:
        curve_computed = curve or self.compute()
        # reorder: recall along x-axis, precision along y-axis
        curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])
        score = _reduce_auroc(curve_computed[0], curve_computed[1], average=None, direction=-1.0)
        return score[1]
    else:    
        curve_computed = curve or self.compute()
        # reorder: recall along x-axis, precision along y-axis
        curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])
        score = _reduce_auroc(curve_computed[0], curve_computed[1], average=None, direction=-1.0)
        return score.mean()

MulticlassPrecisionRecallCurve.compute_AUPR = compute_AUPR

# -------------------------
# 2. Define metrics mapping and compute_metrics
# -------------------------
metrics = {
    "ACC": MulticlassAccuracy,
    "AUROC": MulticlassAUROC,
    "F1": MulticlassF1Score,
    "AUPR": MulticlassPrecisionRecallCurve,
}

def compute_metrics(preds_tensor, targets_tensor):
    """
    Compute multiple classification metrics.
    """
    num_classes = preds_tensor.shape[1]
    results = {}

    # Accuracy (micro-averaged)
    metric = metrics["ACC"](num_classes=num_classes, average="micro")
    results["ACC"] = metric(preds_tensor, targets_tensor).item()

    # Balanced Accuracy (macro-averaged)
    metric = metrics["ACC"](num_classes=num_classes, average="macro")
    results["BAL_ACC"] = metric(preds_tensor, targets_tensor).item()

    # AUROC (macro-averaged)
    metric = metrics["AUROC"](num_classes=num_classes, average="macro")
    results["AUROC"] = metric(preds_tensor, targets_tensor).item()

    # Weighted F1 Score
    metric = metrics["F1"](num_classes=num_classes, average="weighted")
    results["W_F1"] = metric(preds_tensor, targets_tensor).item()

    # AUPR (mean over classes)
    metric = metrics["AUPR"](num_classes=num_classes)
    metric.update(preds_tensor, targets_tensor)
    results["AUPR"] = metric.compute_AUPR(num_classes=num_classes).item()

    return results

# -------------------------
# 3. Bootstrap confidence intervals
# -------------------------
def bootstrap_ci(preds, targets, n_bootstrap=1000, confidence_level=0.95, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(targets)
    stats = {k: [] for k in ["ACC", "BAL_ACC", "AUROC", "W_F1", "AUPR"]}

    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_bootstrap}")
        
        idx = np.random.choice(n, n, replace=True)
        p_sample = preds[idx].astype(np.float32)
        t_sample = targets[idx].astype(np.int64)
        
        p_tensor = torch.from_numpy(p_sample)
        t_tensor = torch.from_numpy(t_sample)
        
        res = compute_metrics(p_tensor, t_tensor)
        for k in stats.keys():
            stats[k].append(res[k])

    ci_dict = {}
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    for k, values in stats.items():
        values = np.array(values)
        mean = np.mean(values)
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        ci_dict[k] = (mean, lower, upper)
    
    return ci_dict

# -------------------------
# 4. NEW: Batch evaluate all preds_*.npy in a directory
# -------------------------
def evaluate_test_in_dir(output_dir: str, n_bootstrap: int = 1000):
    """
    Only evaluate preds_*_TEST.npy (no _0/_1/_2 suffix).
    Requires corresponding target_*_TEST.npy.
    Saves result as eval_{suffix}.csv.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    # 查找所有 preds_*_TEST.npy 文件
    pred_files = list(output_path.glob("preds_*_TEST.npy"))
    
    if not pred_files:
        print("No preds_*_TEST.npy files found.")
        return

    # 取第一个（通常只有一个）
    pred_file = pred_files[0]
    suffix = pred_file.name[len("preds_"):-len(".npy")]  # e.g., "HIBA_10_TEST"
    target_file = output_path / f"target_{suffix}.npy"

    if not target_file.exists():
        raise FileNotFoundError(f"Corresponding target file not found: {target_file}")

    print(f"\nEvaluating: {pred_file.name}")
    
    preds = np.load(pred_file)
    targets = np.load(target_file)

    if targets.dtype != np.int64:
        targets = targets.astype(np.int64)

    if preds.max() > 1.0 or preds.min() < 0:
        print("Converting logits to probabilities using softmax...")
        preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()

    try:
        results_ci = bootstrap_ci(preds, targets, n_bootstrap=n_bootstrap)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Save as eval_{suffix}.csv
    csv_path = output_path / f"eval_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "CI_Lower", "CI_Upper"])
        for metric, (mean, low, up) in results_ci.items():
            writer.writerow([metric, f"{mean:.4f}", f"{low:.4f}", f"{up:.4f}"])

    print(f"Saved evaluation to {csv_path}")
# -------------------------
# 5. Main execution logic
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results with bootstrap CI.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing preds_*.npy and target_*.npy")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate ALL preds_*.npy files (default: only preds_*_TEST.npy)")
    parser.add_argument("--pred_file", type=str, help="Path to specific preds file")
    parser.add_argument("--target_file", type=str, help="Path to specific target file")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()

    if args.pred_file and args.target_file:
        # 单文件模式（不变）
        preds = np.load(args.pred_file)
        targets = np.load(args.target_file)
        print(f"Predictions shape: {preds.shape}, dtype: {preds.dtype}")
        print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")

        if preds.max() > 1.0 or preds.min() < 0:
            print("Converting logits to probabilities using softmax...")
            preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()

        preds_tensor = torch.from_numpy(preds).float()
        targets_tensor = torch.from_numpy(targets).long()

        print("\n" + "="*48)
        print("Computing bootstrap confidence intervals...")
        print("="*48 + "\n")

        results_ci = bootstrap_ci(preds, targets, n_bootstrap=args.n_bootstrap)

        print("\n" + "="*48)
        print("Bootstrap Results (95% CI):")
        print("="*48)
        for k, (mean, lower, upper) in results_ci.items():
            print(f"{k:10s}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
        print("="*48 + "\n")

        base_dir = os.path.dirname(args.pred_file)
        csv_name = os.path.basename(args.pred_file).replace("preds_", "bootstrap_results_").replace(".npy", ".csv")
        csv_path = os.path.join(base_dir, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "CI_Lower", "CI_Upper"])
            for metric, (mean, lower, upper) in results_ci.items():
                writer.writerow([metric, f"{mean:.4f}", f"{lower:.4f}", f"{upper:.4f}"])
        print(f"Results saved to {csv_path}")

    elif args.all:
        # 批量评估所有文件
        # evaluate_all_in_dir(args.output_dir, n_bootstrap=args.n_bootstrap)
        pass

    else:
        # 默认行为：只评估最后一个 TEST 文件
        evaluate_test_in_dir(args.output_dir, n_bootstrap=args.n_bootstrap)