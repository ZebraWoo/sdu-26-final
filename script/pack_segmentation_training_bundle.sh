#!/usr/bin/env bash
# =============================================================================
# 打包：ROSE 语义分割训练（HAM10k + Mask2Former 等）所需代码与根目录配置。
# 大数据（HAM10000 图像/掩膜、teacher SSL 权重）请单独传输，见 SEGMENTATION_TRANSFER_CHECKLIST.txt
#
# 用法：
#   chmod +x script/pack_segmentation_training_bundle.sh
#   ./script/pack_segmentation_training_bundle.sh
#   OUT_DIR=/data/tmp OUT_NAME=my_bundle.tar.gz ./script/pack_segmentation_training_bundle.sh
#
# 可选：去掉与分割无关的 eval 子树（省空间；分割训练不 import 它们）
#   TRIM_EVAL_EXTRAS=1 ./script/pack_segmentation_training_bundle.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_NAME="${OUT_NAME:-rose-seg-training-bundle-${STAMP}.tar.gz}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT}"
ARCHIVE="$OUT_DIR/$OUT_NAME"
TRIM_EVAL_EXTRAS="${TRIM_EVAL_EXTRAS:-0}"

echo "Repo: $REPO_ROOT"
echo "Archive: $ARCHIVE"

TMP_LIST="$(mktemp)"
cleanup() { rm -f "$TMP_LIST"; }
trap cleanup EXIT

find rose -type f \
  ! -path '*/__pycache__/*' \
  ! -name '*.pyc' \
  ! -path '*/.mypy_cache/*' \
  ! -name '.DS_Store' \
  \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.cpp' -o -name '*.cu' \
     -o -name '*.h' -o -name '*.cuh' -o -name '*.md' -o -name 'setup.py' -o -name 'Makefile' \
     -o -name '*.txt' -o -name '*.toml' -o -name '*.json' \) | sort -u >"$TMP_LIST"

if [[ "$TRIM_EVAL_EXTRAS" == "1" ]]; then
  grep -vE '^rose/eval/detection/|^rose/eval/dense/|^rose/eval/text/' "$TMP_LIST" >"${TMP_LIST}.f" && mv "${TMP_LIST}.f" "$TMP_LIST"
  echo "(trimmed eval/detection, eval/dense, eval/text)"
fi

{
  for f in \
    setup.py \
    requirements.txt \
    requirements-dev.txt \
    pyproject.toml \
    config.yaml \
    eval_config.yaml \
    LICENSE.md \
    README.md \
    hubconf.py \
    script/pack_segmentation_training_bundle.sh \
    script/SEGMENTATION_TRANSFER_CHECKLIST.txt
  do
    [[ -f "$f" ]] && echo "$f"
  done
} >>"$TMP_LIST"

sort -u "$TMP_LIST" -o "$TMP_LIST"

tar -czf "$ARCHIVE" -C "$REPO_ROOT" -T "$TMP_LIST"
echo "Done. Size: $(du -h "$ARCHIVE" | cut -f1)"

echo ""
echo "目标机建议："
echo "  mkdir -p ~/proj && tar -xzf $OUT_NAME -C ~/proj"
echo "  cd ~/proj && pip install -r requirements.txt"
echo "  pip install -e .   # 若失败可用: export PYTHONPATH=\"\$(pwd)\""
echo "  若需编译 MSDeformAttn: cd rose/eval/segmentation/models/utils/ops && pip install -e ."
