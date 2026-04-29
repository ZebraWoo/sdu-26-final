#!/usr/bin/env bash
# HAM10K + ROSE + Mask2Former 分割训练（多卡）
# 用法：bash rose/eval/segmentation/run_ham10k_m2f_seg.sh
# 先修改下方「配置区」路径；若不用 conda，可注释掉 conda 两行。

set -euo pipefail
cd "$(dirname "$0")/../../.."  # 项目根（与 run.py 中 PROJECT_ROOT 一致）
echo "WORKDIR: $(pwd)"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# ---------- 配置区（按本机修改）----------
CONFIG_YAML="rose/eval/segmentation/configs/config-ham10k-m2f-seg-training-v1.yaml"
# 若沿用旧配置可改为：config-ham10k-m2f-seg-training-v1.yaml

MODEL_CONFIG_FILE="/home/wuzuoxu/Data/skin_dataset/config.yaml"
MODEL_PRETRAINED_WEIGHTS="/home/wuzuoxu/Data/skin_dataset/training_35999/teacher_checkpoint.pth"
DATASETS_ROOT="/home/wuzuoxu/Data/skin_dataset/ssl/HAM10000"
OUTPUT_DIR="/home/wuzuoxu/Data/skin_dataset/rose-outputs/segmentation/ham10k_m2f_train_matching_v4"

# 指定可用 GPU 列表（可覆盖：GPU_LIST=4,5,6,7 bash ...）
GPU_LIST="${GPU_LIST:-4,5,6,7}"
# 每卡 batch size（可覆盖：BS=1 bash ...）
BS="${BS:-1}"
# 自动按 GPU 列表得到卡数
N_GPUS="$(awk -F',' '{print NF}' <<< "${GPU_LIST}")"
# 显存紧张时保持 BS=1；对比实验请固定 BS
# ------------------------------------------

# 可选：激活 conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null) || true
  if [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate rose
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (n_gpus=${N_GPUS}, bs=${BS})"

exec torchrun --nproc_per_node="${N_GPUS}" -m rose.eval.segmentation.run \
  "config=${CONFIG_YAML}" \
  "model.config_file=${MODEL_CONFIG_FILE}" \
  "model.pretrained_weights=${MODEL_PRETRAINED_WEIGHTS}" \
  "datasets.root=${DATASETS_ROOT}" \
  "output_dir=${OUTPUT_DIR}" \
  "n_gpus=${N_GPUS}" \
  "bs=${BS}"
