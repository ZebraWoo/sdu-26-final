export PYTHONPATH=${PWD}

python -m pdb -m dinov3.run.submit dinov3/train/train.py \
  --nodes 1 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir /sdu/haodi/imagenet1k-256/dinov3_outputs \
  train.dataset_path=ImageNet:split=TRAIN:root=/sdu/haodi/imagenet1k-256:extra=/sdu/haodi/imagenet1k-256/extra