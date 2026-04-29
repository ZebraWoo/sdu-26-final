export PYTHONPATH="${PWD}:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/eval/linear.py \
  model.model_name=ROSE \
  model.config_file=/data/wenjing/skin_dataset/rose-outputs/config.yaml \
  model.pretrained_weights=/data/wenjing/skin_dataset/rose-outputs/eval/training_35999/teacher_checkpoint.pth \
  output_dir=/data/wenjing/skin_dataset/rose-outputs/reader_study \
  train.dataset=HAM10K:split=TRAIN:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  train.val_dataset=HAM10K:split=VAL:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra \
  eval.test_datasets=["HAM10K:split=TEST:root=/data/wenjing/skin_dataset/ssl:extra=/data/wenjing/skin_dataset/ssl/extra"] \