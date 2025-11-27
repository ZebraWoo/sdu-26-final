import numpy as np
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
)
import torch
from torch import randn, randint
from torchmetrics.functional.classification.auroc import _reduce_auroc

import torch
import numpy as np
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
)
from torchmetrics.functional.classification.auroc import _reduce_auroc

# -------------------------
# 1. 扩展 MulticlassPrecisionRecallCurve 增加 compute_AUPR 方法
# -------------------------
def compute_AUPR(self, curve=None):
    """Return AUPR (mean over classes), without plotting."""
    curve_computed = curve or self.compute()
    # reorder: recall along x-axis, precision along y-axis
    curve_computed = (curve_computed[1], curve_computed[0], curve_computed[2])
    score = _reduce_auroc(curve_computed[0], curve_computed[1], average=None, direction=-1.0)
    return score.mean()

MulticlassPrecisionRecallCurve.compute_AUPR = compute_AUPR

# -------------------------
# 2. 加载预测结果和标签
# -------------------------
preds = np.load("/data/wenjing/skin_dataset/DermDino-3-outputs-1015/linear_output/preds_HAM10K_TEST.npy")
targets = np.load("/data/wenjing/skin_dataset/DermDino-3-outputs-1015/linear_output/target_HAM10K_TEST.npy")
preds_tensor = torch.from_numpy(preds)
targets_tensor = torch.from_numpy(targets)

# -------------------------
# 3. 定义 metrics 映射
# -------------------------
metrics = {
    "ACC": MulticlassAccuracy,
    "AUROC": MulticlassAUROC,
    "F1": MulticlassF1Score,
    "AUPR": MulticlassPrecisionRecallCurve,
}

# -------------------------
# 4. 定义一个函数计算各项指标
# -------------------------
def compute_metrics(preds_tensor, targets_tensor):
    num_classes = preds_tensor.shape[1]
    results = {}

    # Average Accuracy
    metric = metrics["ACC"](num_classes=num_classes, average="micro")
    results["ACC"] = metric(preds_tensor, targets_tensor).item()

    # Balanced Accuracy
    metric = metrics["ACC"](num_classes=num_classes, average="macro")
    results["BAL_ACC"] = metric(preds_tensor, targets_tensor).item()

    # AUROC
    metric = metrics["AUROC"](num_classes=num_classes, average="macro")
    results["AUROC"] = metric(preds_tensor, targets_tensor).item()

    # Weighted F1
    metric = metrics["F1"](num_classes=num_classes, average="weighted")
    results["W_F1"] = metric(preds_tensor, targets_tensor).item()

    # AUPR
    metric = metrics["AUPR"](num_classes=num_classes)
    metric.update(preds_tensor, targets_tensor)
    results["AUPR"] = metric.compute_AUPR().item()

    return results

# -------------------------
# 5. 定义 Bootstrap 置信区间计算
# -------------------------
def bootstrap_ci(preds, targets, n_bootstrap=1000, confidence_level=0.95, seed=42):
    np.random.seed(seed)
    n = len(targets)
    stats = {k: [] for k in ["ACC", "BAL_ACC", "AUROC", "W_F1", "AUPR"]}

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        p_sample = preds[idx]
        t_sample = targets[idx]
        res = compute_metrics(torch.from_numpy(p_sample), torch.from_numpy(t_sample))
        for k in stats.keys():
            stats[k].append(res[k])

    ci_dict = {}
    for k, values in stats.items():
        lower = np.percentile(values, ((1 - confidence_level) / 2) * 100)
        upper = np.percentile(values, (1 - (1 - confidence_level) / 2) * 100)
        ci_dict[k] = (np.mean(values), lower, upper)
    return ci_dict

# -------------------------
# 6. 执行计算
# -------------------------
results_ci = bootstrap_ci(preds, targets, n_bootstrap=1000)
for k, (mean, lower, upper) in results_ci.items():
    print(f"{k}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
