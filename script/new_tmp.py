import numpy as np
import random
from PIL import Image
import torch
import sys
PROJECT_ROOT = "/home/jiangwenjing/hd/dinov3"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rose.data.masking import MaskingGenerator, RandomMaskingGenerator

# 加载并调整图像
image_path = "/data/wenjing/skin_dataset/ssl/ssl_data/bcn20000/ISIC_0053677.jpg"
image = Image.open(image_path)
image = image.resize((224, 224))  # 调整图像尺寸以适应掩码生成器
image_np = np.array(image)

# 训练设置
img_size = 224  # 图像大小
patch_size = 16  # 每个 patch 的大小
n_tokens = (img_size // patch_size) ** 2  # 图像划分为的小块数
random_circular_shift = False  # 是否进行随机圆形平移

# 创建掩码生成器
mask_generator = MaskingGenerator(
    input_size=(img_size // patch_size, img_size // patch_size),
    max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
)
'''mask_generator = RandomMaskingGenerator(
    input_size=(img_size // patch_size, img_size // patch_size)
    )'''

mask = torch.BoolTensor(mask_generator(int(n_tokens * 0.5)))

# 创建一个空白的图像矩阵以保存所有掩盖的区域
final_masked_image = image_np.copy()

mask = mask.numpy().reshape(img_size // patch_size, img_size // patch_size)

# 将掩码位置的像素设为黑色
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j]:
            top_left_x = j * patch_size
            top_left_y = i * patch_size
            # 确保索引不超出图像的尺寸
            final_masked_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size] = 0

# 如果是RGB图像，确保图像通道处理正确
if final_masked_image.shape[-1] == 3:
    final_masked_image = np.array(final_masked_image, dtype=np.uint8)

# 将最终的掩盖图像保存为 masked_image_all_masks.jpg
final_masked_image_pil = Image.fromarray(final_masked_image)
final_masked_image_pil.save("script/test/random_masked_image_all_masks.jpg")

# 可选：如果你想查看最终的掩盖效果，可以取消注释以下代码：
# import matplotlib.pyplot as plt
# plt.imshow(final_masked_image)
# plt.axis("off")
# plt.show()
