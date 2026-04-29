export PYTHONPATH="${PWD}:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/eval/linear.py \
  model.model_name=PanDerm \
  model.config_file=None \
  model.pretrained_weights=/data/wenjing/skin_dataset/panderm_eval/panderm_ll_data6_checkpoint-499.pth \
  output_dir=/data/wenjing/skin_dataset/panderm_eval/fewshot/ham_80 \
  train.dataset=HAM10K:split=TRAIN:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  train.val_dataset=HAM10K:split=VAL:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  train.avg=False \
  eval.test_datasets=["HAM10K:split=TEST:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra"]