export PYTHONPATH="${PWD}:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rose/eval/segmentation/test.py \
  model.model_name=PanDerm \
  model.config_file=None \
  model.pretrained_weights=/data/wenjing/skin_dataset/panderm_eval/panderm_ll_data6_checkpoint-499.pth \
  config=rose/eval/segmentation/configs/config-ham10k-linear-seg-training.yaml \
  datasets.root=/data/wenjing/skin_dataset/ssl/segmentation/HAM10000 \
  load_from=/data/wenjing/skin_dataset/panderm_eval/segmentation/ham10k_panderm_linear_seg_output/model_final.pth \
  output_dir=/data/wenjing/skin_dataset/panderm_eval/segmentation/ham10k_panderm_linear_seg_output