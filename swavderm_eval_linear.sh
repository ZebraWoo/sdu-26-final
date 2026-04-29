export PYTHONPATH="${PWD}:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/eval/linear.py \
  model.model_name=SwAVDerm \
  model.config_file=None \
  model.pretrained_weights=/data/wenjing/skin_dataset/swavderm_eval/derm_pretrained.pth \
  output_dir=/data/wenjing/skin_dataset/swavderm_eval/fewshot/ham_10 \
  train.dataset=HAM10K:split=TRAIN:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  train.val_dataset=HAM10K:split=VAL:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  train.avg=False \
  eval.test_datasets=["HAM10K:split=TEST:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra"] \
  