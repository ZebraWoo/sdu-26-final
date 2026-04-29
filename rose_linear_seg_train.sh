export PYTHONPATH="${PWD}:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/eval/segmentation/run.py \
  model.model_name=ROSE \
  model.config_file=/data/wenjing/skin_dataset/rose-outputs/config.yaml \
  model.pretrained_weights=/data/wenjing/skin_dataset/rose-outputs/eval/training_35999/teacher_checkpoint.pth \
  config=rose/eval/segmentation/configs/config-ham10k-linear-seg-training.yaml \
  datasets.root=/data/wenjing/skin_dataset/ssl/segmentation/HAM10000 \
  output_dir=/data/wenjing/skin_dataset/rose-outputs/segmentation/ham10k_rose_linear_seg_output \