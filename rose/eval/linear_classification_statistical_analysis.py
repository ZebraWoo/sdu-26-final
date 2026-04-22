import os
import numpy as np
import torch
from scipy.stats import ttest_rel
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
)
from torchmetrics.functional.classification.auroc import _reduce_auroc

# ============================================================
# 0. Global configuration
# ============================================================
ROOT = "/data/wenjing/skin_dataset"

BASELINE_MODEL = "rose-outputs"
COMPARE_MODELS = ["panderm_eval", "swavderm_eval"]

SUBFOLDERS = {
    "HAM10K": "ham_linear_output",
    "BCN20K": "bcn20k_linear_output",
    "HIBA": "hiba_linear_output",
    "MNDICHD": "mn_linear_output",
}

PRED_FILES = {
    "HAM10K": "preds_HAM10K_TEST.npy",
    "BCN20K": "preds_BCN20K_TEST.npy",
    "HIBA": "preds_HIBA_TEST.npy",
    "MNDICHD": "preds_MNDICHD_TEST.npy",
}

TARGET_FILES = {
    "HAM10K": "target_HAM10K_TEST.npy",
    "BCN20K": "target_BCN20K_TEST.npy",
    "HIBA": "target_HIBA_TEST.npy",
    "MNDICHD": "target_MNDICHD_TEST.npy",
}

N_BOOTSTRAP = 1000
SEED = 42


# ============================================================
# 1. AUPR definition (exactly aligned with your code)
# ============================================================
def compute_AUPR(self, num_classes=None, curve=None):
    curve = curve or self.compute()
    curve = (curve[1], curve[0], curve[2])
    score = _reduce_auroc(curve[0], curve[1], average=None, direction=-1.0)
    if num_classes == 2:
        return score[1]
    return score.mean()


MulticlassPrecisionRecallCurve.compute_AUPR = compute_AUPR


# ============================================================
# 2. Metric computation
# ============================================================
def compute_metrics(preds_tensor, targets_tensor):
    num_classes = preds_tensor.shape[1]
    res = {}

    res["ACC"] = MulticlassAccuracy(num_classes, average="micro")(
        preds_tensor, targets_tensor
    ).item()

    res["BAL_ACC"] = MulticlassAccuracy(num_classes, average="macro")(
        preds_tensor, targets_tensor
    ).item()

    res["AUROC"] = MulticlassAUROC(num_classes, average="macro")(
        preds_tensor, targets_tensor
    ).item()

    res["W_F1"] = MulticlassF1Score(num_classes, average="weighted")(
        preds_tensor, targets_tensor
    ).item()

    pr = MulticlassPrecisionRecallCurve(num_classes)
    pr.update(preds_tensor, targets_tensor)
    res["AUPR"] = pr.compute_AUPR(num_classes).item()

    return res


# ============================================================
# 3. Load preds & targets
# ============================================================
def load_preds_targets(model, dataset):
    folder = SUBFOLDERS[dataset]
    preds_path = os.path.join(ROOT, model, folder, PRED_FILES[dataset])
    targets_path = os.path.join(ROOT, model, folder, TARGET_FILES[dataset])

    preds = np.load(preds_path)
    targets = np.load(targets_path)

    # logits → probability
    if preds.max() > 1.0 or preds.min() < 0:
        preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()

    return preds.astype(np.float32), targets.astype(np.int64)


# ============================================================
# 4. Bootstrap + paired t-test
# ============================================================
def run_stat_test(rose_data, other_data, dataset, model_name):
    np.random.seed(SEED)
    N = len(rose_data["targets"])

    metric_names = ["ACC", "BAL_ACC", "AUROC", "W_F1", "AUPR"]
    rose_stats = {m: [] for m in metric_names}
    other_stats = {m: [] for m in metric_names}

    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(N, N, replace=True)

        r_preds = torch.from_numpy(rose_data["preds"][idx]).float()
        r_tgts = torch.from_numpy(rose_data["targets"][idx]).long()
        o_preds = torch.from_numpy(other_data["preds"][idx]).float()
        o_tgts = torch.from_numpy(other_data["targets"][idx]).long()

        r_res = compute_metrics(r_preds, r_tgts)
        o_res = compute_metrics(o_preds, o_tgts)

        for m in metric_names:
            rose_stats[m].append(r_res[m])
            other_stats[m].append(o_res[m])

    print(f"\n[{dataset}]  {model_name}  vs  ROSE")
    print("-" * 90)

    for m in metric_names:
        r = np.array(rose_stats[m])
        o = np.array(other_stats[m])

        t, p = ttest_rel(o, r)

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        print(
            f"{m:8s} | "
            f"ROSE {r.mean():.4f}±{r.std():.4f} | "
            f"{model_name} {o.mean():.4f}±{o.std():.4f} | "
            f"t={t:7.3f}, p={p:.2e} {sig}"
        )


# ============================================================
# 5. Main entry
# ============================================================
def main():
    for dataset in SUBFOLDERS.keys():
        rose_preds, rose_targets = load_preds_targets(BASELINE_MODEL, dataset)
        rose_data = {"preds": rose_preds, "targets": rose_targets}

        for model in COMPARE_MODELS:
            other_preds, other_targets = load_preds_targets(model, dataset)

            assert np.array_equal(
                rose_targets, other_targets
            ), f"GT mismatch in {dataset}!"

            other_data = {"preds": other_preds, "targets": other_targets}

            run_stat_test(rose_data, other_data, dataset, model)


if __name__ == "__main__":
    main()
