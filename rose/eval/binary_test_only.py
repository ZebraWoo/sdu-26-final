import numpy as np
import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecisionRecallCurve,
)
from torchmetrics.utilities.compute import _auc_compute_without_check

# -------------------------
# 1. Extend BinaryPrecisionRecallCurve to compute AUPR
# -------------------------
def compute_AUPR(self, curve=None):
    """Return AUPR (Area Under Precision-Recall Curve)."""
    curve_computed = curve or self.compute()
    # reorder: recall along x-axis, precision along y-axis
    curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])
    score = _auc_compute_without_check(curve_computed[0], curve_computed[1], direction=-1.0)
    return score

BinaryPrecisionRecallCurve.compute_AUPR = compute_AUPR

# -------------------------
# 2. Load predictions and targets
# -------------------------
preds = np.load("/data/wenjing/skin_dataset/panderm_eval/mn_linear_output/preds_MNDICHD_TEST.npy")
targets = np.load("/data/wenjing/skin_dataset/panderm_eval/mn_linear_output/target_MNDICHD_TEST.npy")

# 验证数据类型和形状
print(f"Predictions shape: {preds.shape}, dtype: {preds.dtype}")
print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")

if preds.shape[1] == 2:
    print("Converting from (N, 2) logits/probabilities to (N,) format...")
    if preds.max() > 1.0 or preds.min() < 0:
        preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()
    preds = preds[:, 1]
else:
    print(f"Warning: Unexpected shape {preds.shape}. Expected (N,) or (N, 2)")
    preds = preds.squeeze()

# 确保是正确的形状和类型
preds_tensor = torch.from_numpy(preds).float()
targets_tensor = torch.from_numpy(targets).long()

print(f"Final predictions shape: {preds_tensor.shape}, dtype: {preds_tensor.dtype}")
print(f"Final targets shape: {targets_tensor.shape}, dtype: {targets_tensor.dtype}")

# -------------------------
# 3. Define metrics mapping
# -------------------------
metrics = {
    "ACC": BinaryAccuracy,
    "AUROC": BinaryAUROC,
    "F1": BinaryF1Score,
    "AUPR": BinaryPrecisionRecallCurve,
}

# -------------------------
# 4. Define metric computation function
# -------------------------
def compute_metrics(preds_tensor, targets_tensor):
    """
    Compute multiple binary classification metrics.
    
    Args:
        preds_tensor: shape (N,), probabilities (0-1)
        targets_tensor: shape (N,), binary labels (0 or 1)
    
    Returns:
        dict of metric values
    """
    results = {}

    # Accuracy
    metric = metrics["ACC"]()
    results["ACC"] = metric(preds_tensor, targets_tensor).item()

    # AUROC
    metric = metrics["AUROC"]()
    results["AUROC"] = metric(preds_tensor, targets_tensor).item()

    # F1 Score
    metric = metrics["F1"]()
    results["F1"] = metric(preds_tensor, targets_tensor).item()

    # AUPR (Area Under Precision-Recall Curve)
    metric = metrics["AUPR"]()
    metric.update(preds_tensor, targets_tensor)
    results["AUPR"] = metric.compute_AUPR().item()

    return results

# -------------------------
# 5. Bootstrap confidence intervals
# -------------------------
def bootstrap_ci(preds, targets, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        preds: shape (N,), numpy array of probabilities
        targets: shape (N,), numpy array of binary labels
        n_bootstrap: number of bootstrap samples (default: 1000)
        confidence_level: confidence level (default: 0.95 for 95% CI)
        seed: random seed for reproducibility
    
    Returns:
        dict mapping metric names to (mean, lower, upper) tuples
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(targets)
    stats = {k: [] for k in ["ACC", "AUROC", "F1", "AUPR"]}

    print(f"Running {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{n_bootstrap}")
        
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        p_sample = preds[idx].astype(np.float32)
        t_sample = targets[idx].astype(np.int64)
        
        # Convert to tensors
        p_tensor = torch.from_numpy(p_sample)
        t_tensor = torch.from_numpy(t_sample)
        
        # Compute metrics
        res = compute_metrics(p_tensor, t_tensor)
        for k in stats.keys():
            stats[k].append(res[k])

    # Compute confidence intervals
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
# 6. Compute and print results
# -------------------------
print("\n" + "="*48)
print("Computing bootstrap confidence intervals...")
print("="*48 + "\n")

results_ci = bootstrap_ci(preds, targets, n_bootstrap=1000, confidence_level=0.95)

print("\n" + "="*48)
print("Bootstrap Results (95% CI):")
print("="*48)
for k, (mean, lower, upper) in results_ci.items():
    print(f"{k:10s}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
print("="*48 + "\n")
