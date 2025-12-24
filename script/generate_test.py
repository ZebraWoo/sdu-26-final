import os
import random
import shutil

# 路径
src_dir = "/sdu/haodi/imagenet1k-256/val"
dst_dir = "/sdu/haodi/imagenet1k-256/test"
os.makedirs(dst_dir, exist_ok=True)

# 获取所有 jpeg 文件
jpeg_files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
print(f"在 val 中找到 {len(jpeg_files)} 个 jpg 文件")

# 随机采样 20000 个
random.seed(42)  # 保证可复现
sampled = random.sample(jpeg_files, 20000)

for fname in sampled:
    base, _ = os.path.splitext(fname)  # 去掉后缀
    # 原始路径
    jpeg_src = os.path.join(src_dir, base + ".jpg")
    cls_src  = os.path.join(src_dir, base + ".cls")
    json_src = os.path.join(src_dir, base + ".json")

    # 新文件名，把 val 改为 test
    new_base = base.replace("val", "test")
    jpeg_dst = os.path.join(dst_dir, new_base + ".jpg")
    cls_dst  = os.path.join(dst_dir, new_base + ".cls")
    json_dst = os.path.join(dst_dir, new_base + ".json")

    # 移动三个文件
    for src, dst in [(jpeg_src, jpeg_dst), (cls_src, cls_dst), (json_src, json_dst)]:
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"[WARN] 缺少伴随文件: {src}")

print("✅ 已完成文件采样和移动")
