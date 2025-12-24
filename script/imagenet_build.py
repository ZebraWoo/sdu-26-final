import os
import tarfile
from multiprocessing import Pool, cpu_count

# 输入输出路径
src_dir = "/sdu/haodi/imagenet1k-256-wds"
dst_dir = "/sdu/haodi/imagenet1k-256"

# train 和 val 的目标路径
train_dir = os.path.join(dst_dir, "train")
val_dir = os.path.join(dst_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


def extract_tar(tar_path):
    """解压单个 tar 文件到对应子目录"""
    fname = os.path.basename(tar_path)
    if "train" in fname.lower():
        out_dir = train_dir
    elif "val" in fname.lower() or "validation" in fname.lower():
        out_dir = val_dir
    else:
        print(f"[WARN] {fname} 不包含 train/validation 关键字，跳过。")
        return

    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=out_dir)
        print(f"[OK] {fname} 解压完成 → {out_dir}")
    except Exception as e:
        print(f"[ERROR] 解压 {fname} 出错: {e}")


def main():
    # 找到所有 tar 文件
    tar_files = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith(".tar")
    ]

    print(f"共找到 {len(tar_files)} 个 tar 文件，开始解压...")

    # 多进程池
    nproc = min(cpu_count(), 24)  # 限制最多 8 个进程，避免 I/O 过载
    with Pool(processes=nproc) as pool:
        pool.map(extract_tar, tar_files)


if __name__ == "__main__":
    main()
