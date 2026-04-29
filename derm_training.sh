export PYTHONPATH="${PWD}:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/train/train.py \
  --config-file rose/configs/train/vitl_derm_pretrain.yaml \
  --output-dir /data/wenjing/skin_dataset/rose-outputs \
  train.dataset_path=Dermoscopy:split=TRAIN:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra