import sys
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
PROJECT_ROOT = "/home/jiangwenjing/hd/rose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rose.data.augmentations import DataAugmentationDerm
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

# ------------------------
# 1. 加载配置
# ------------------------
def get_default_config():
    p = os.path.join(PROJECT_ROOT, "rose", "configs", "ssl_default_config.yaml")
    return OmegaConf.load(p)


def get_configs():
    default_cfg = get_default_config()
    cfg_path = os.path.join(PROJECT_ROOT, "rose", "configs", "train", "vitl_derm_pretrain.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


# ------------------------
# 2. 测试用主函数
# ------------------------
if __name__ == "__main__":
    # 载入配置
    cfg = get_configs()

    # 创建数据增强器
    transform = DataAugmentationDerm(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=cfg.crops.rgb_mean,
            std=cfg.crops.rgb_std,
        )

    # ------------------------
    # 3. 载入一张测试图片
    # ------------------------
    test_image_path = "/data/wenjing/skin_dataset/ssl/ssl_data/challenge2020/ISIC_0175364.jpg"  # 修改成你的图片路径
    image = Image.open(test_image_path).convert("RGB")
    # ------------------------
    # 4. 执行增强
    # ------------------------
    output = transform(image)

    # global crops 是两个视图
    global_crops = output["global_crops"]
    print(f"Global crops generated: {len(global_crops)}")
    local_crops = output["local_crops"]
    print(f"length of local crops: {len(local_crops)}")
    # ------------------------
    # 5. 保存增强结果
    # ------------------------
    save_dir = pathlib.Path("script") / "regularized_output"
    save_dir.mkdir(exist_ok=True)

    for i, crop in enumerate(global_crops):
        # crop 是 Tensor (C,H,W)，需要转回 PIL 图像
        pil_img = TF.to_pil_image(torch.clamp(crop, 0, 1))  # 防止溢出
        save_path = save_dir / f"global_random_{i+1}.jpg"
        pil_img.save(save_path)
        print(f"Saved {save_path}")

    for i, crops in enumerate(local_crops):
        pil_img = TF.to_pil_image(torch.clamp(crops, 0, 1))
        save_path = save_dir / f"local_random_{i+1}.jpg"
        pil_img.save(save_path)
        print(f"saved {save_path}")
