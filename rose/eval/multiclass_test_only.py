import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
)
from torchmetrics.functional.classification.auroc import _reduce_auroc

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
# 2. Load predictions and targets
# -------------------------
preds = np.load("/data/wenjing/skin_dataset/swavderm_eval/ham_linear_output/preds_HAM10K_TEST.npy")
targets = np.load("/data/wenjing/skin_dataset/swavderm_eval/ham_linear_output/target_HAM10K_TEST.npy")

# 验证数据类型和形状
print(f"Predictions shape: {preds.shape}, dtype: {preds.dtype}")
print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")

# 如果preds是logits，需要转换为概率
if preds.max() > 1.0 or preds.min() < 0:
    print("Converting logits to probabilities using softmax...")
    preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()

preds_tensor = torch.from_numpy(preds).float()
targets_tensor = torch.from_numpy(targets).long()

# -------------------------
# 3. Define metrics mapping
# -------------------------
metrics = {
    "ACC": MulticlassAccuracy,
    "AUROC": MulticlassAUROC,
    "F1": MulticlassF1Score,
    "AUPR": MulticlassPrecisionRecallCurve,
}

# -------------------------
# 4. Define metric computation function
# -------------------------
def compute_metrics(preds_tensor, targets_tensor):
    """
    Compute multiple classification metrics.
    
    Args:
        preds_tensor: shape (N, num_classes), probabilities or logits
        targets_tensor: shape (N,), integer labels
    
    Returns:
        dict of metric values
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
# 5. Bootstrap confidence intervals
# -------------------------
def bootstrap_ci(preds, targets, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        preds: shape (N, num_classes), numpy array of probabilities
        targets: shape (N,), numpy array of labels
        n_bootstrap: number of bootstrap samples (default: 1000)
        confidence_level: confidence level (default: 0.95 for 95% CI)
        seed: random seed for reproducibility
    
    Returns:
        dict mapping metric names to (mean, lower, upper) tuples
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(targets)
    stats = {k: [] for k in ["ACC", "BAL_ACC", "AUROC", "W_F1", "AUPR"]}

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

