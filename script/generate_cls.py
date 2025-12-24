import os
import json
import shutil

root_dir = "/sdu/haodi/imagenet1k-256"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")

# ====================
# 处理训练集
# ====================
for file in os.listdir(train_dir):
    if file.endswith(".jpg"):
        file_base = os.path.splitext(file)[0]
        parts = file_base.split("_")
        folder_name = parts[0] if parts else "unknown"
        folder_path = os.path.join(train_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 移动 jpg, cls, json
        for ext in [".jpg", ".cls", ".json"]:
            src_path = os.path.join(train_dir, file_base + ext)
            if os.path.exists(src_path):
                dst_path = os.path.join(folder_path, file_base + ext)
                shutil.move(src_path, dst_path)

# ====================
# 处理验证集
# ====================
for file in os.listdir(val_dir):
    if file.endswith(".json"):
        json_path = os.path.join(val_dir, file)
        with open(json_path, "r") as f:
            data = json.load(f)
            filename_field = data.get("filename", "")
            if not filename_field:
                continue
            folder_name = filename_field.split("/")[0]
        
        folder_path = os.path.join(val_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        file_base = os.path.splitext(file)[0]
        # 移动 jpg, cls, json
        for ext in [".jpg", ".cls", ".json"]:
            src_path = os.path.join(val_dir, file_base + ext)
            if os.path.exists(src_path):
                dst_path = os.path.join(folder_path, file_base + ext)
                shutil.move(src_path, dst_path)

print("训练集和验证集文件整理完成！")
