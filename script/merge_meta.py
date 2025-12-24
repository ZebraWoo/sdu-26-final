import os
import glob
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 参数
# ----------------------------------------------------------
csv_dir = r"/data/wenjing/skin_dataset/ssl/ssl_data"
npy_dir = r"/data/wenjing/skin_dataset/ssl/extra"
save_path = r"/data/wenjing/skin_dataset/ssl/merged_ssl.csv"

# 需要保留的字段
keep_cols = [
    "isic_id",
    "attribution",
    "copyright_license",
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_5",
    "image_type"
]

# ----------------------------------------------------------
# Step 1: 读取所有 CSV 并按字段合并 + 记录来源文件名
# ----------------------------------------------------------
all_csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
df_list = []

print("读取 CSV 文件数量:", len(all_csv_files))

for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)

    # 补齐缺失字段
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 添加来源数据集名称
    dataset_name = csv_file.split("/")[-1].split(".")[0]
    df["source_dataset"] = os.path.basename(dataset_name)   # 例如 "HAM10000_metadata.csv"

    df_list.append(df[keep_cols + ["source_dataset"]])

# 合并
merged = pd.concat(df_list, ignore_index=True)

# 按 isic_id 去重，只保留第一条
merged = merged.drop_duplicates(subset=["isic_id"], keep="first")

# 过滤 image_type
merged = merged[merged["image_type"] == "dermoscopic"]

print("CSV 合并后数量（仅 dermoscopic）:", len(merged))

# ----------------------------------------------------------
# Step 2: 读取 *-TEST.npy 和 ham*.npy，提取 image_id 进行过滤
# ----------------------------------------------------------
test_npy_files = glob.glob(os.path.join(npy_dir, "*-TEST.npy"))
test_npy_files += glob.glob(os.path.join(npy_dir, "ham*.npy"))

print("读取到 NPY 文件数量:", len(test_npy_files))

remove_ids = set()

for npy_file in test_npy_files:
    arr = np.load(npy_file, allow_pickle=True)

    # dict
    if isinstance(arr, dict):
        if "image_id" in arr:
            remove_ids.update(arr["image_id"].tolist())

    # structured array 或 array
    elif isinstance(arr, np.ndarray):
        if arr.dtype.names and "image_id" in arr.dtype.names:
            remove_ids.update(arr["image_id"].tolist())
        else:
            # list of dict
            try:
                for item in arr.tolist():
                    if isinstance(item, dict) and "image_id" in item:
                        remove_ids.add(item["image_id"])
            except Exception:
                pass

print("从 npy 中收集到要删除的 image_id 数量:", len(remove_ids))

# ----------------------------------------------------------
# Step 3: 删除 npy 中的样本
# ----------------------------------------------------------
before_count = len(merged)
merged = merged[~merged["isic_id"].isin(remove_ids)]
after_count = len(merged)

print(f"删除 npy 中重复 ID 后：{before_count} -> {after_count}")

# ----------------------------------------------------------
# Step 4: 保存结果
# ----------------------------------------------------------
merged.to_csv(save_path, index=False)
print(f"已保存到: {save_path}")
