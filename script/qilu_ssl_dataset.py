import os
import fitz  # PyMuPDF
import shutil
import io
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor

# 计算图库图片hash
def compute_hash(img_path):
    img = Image.open(img_path)
    return img_path, imagehash.phash(img)

# 路径配置
base_dir = "/sdu/haodi/skin_dataset/qilu_data/"
sub_dirs = [
    base_dir + str(i) + "/" 
    for i in (
        list(range(2309, 2313)) +    # 2309–2312
        list(range(2401, 2413)) +    # 2401–2412
        [b for b in range(2501, 2507) if b not in (2502, 2503)]  # 2501–2506 (排除2502,2503)
    )
]
image_dirs = [sub_dir + "20" + sub_dir.split("/")[-2] for sub_dir in sub_dirs]

pdf_dir = "/sdu/haodi/skin_dataset/qilu_data/2308_2507_reports"
dst_dir = "/sdu/haodi/skin_dataset/qilu_data/ssl_1"
os.makedirs(dst_dir, exist_ok=True)

# 匹配阈值（哈希距离）
DIST_THRESHOLD = 10

# ====================================================
# 1. 预计算所有图库 hash
# ====================================================
print("预计算所有图库 hash...")
all_img_paths = []
for image_dir in image_dirs:
    if not os.path.exists(image_dir):
        print(f"⚠️ 路径不存在: {image_dir}")
        continue
    all_img_paths.extend(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    )

with ProcessPoolExecutor() as executor:
    gallery_hashes = list(executor.map(compute_hash, all_img_paths))

print(f"图库总共 {len(gallery_hashes)} 张图像")

# ====================================================
# 2. 遍历所有 PDF
# ====================================================
for pdf_file in os.listdir(pdf_dir):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_dir, pdf_file)
    print(f"\n处理 {pdf_file} ...")

    # 打开PDF并提取第一页的图片
    doc = fitz.open(pdf_path)
    page = doc[0]
    images = page.get_images(full=True)

    # 跳过 logo/sign
    if len(images) > 2:
        images = images[1:-1]

    pdf_name = os.path.splitext(pdf_file)[0]

    for ind, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        # 转换为PIL.Image对象（不落盘）
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        query_hash = imagehash.phash(img_pil)

        # 在所有图库中找到最佳匹配
        best_path, best_hash = min(gallery_hashes, key=lambda kv: query_hash - kv[1])
        best_dist = query_hash - best_hash

        if best_dist <= DIST_THRESHOLD:
            new_img_name = f"{pdf_name}_{ind}.jpg"
            dst_img_path = os.path.join(dst_dir, new_img_name)
            shutil.copy(best_path, dst_img_path)
            # print(f"  图片{ind}: 匹配 {os.path.basename(best_path)}, 距离={best_dist}, 保存为 {new_img_name}")

    doc.close()
