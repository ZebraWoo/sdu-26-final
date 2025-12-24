import os
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_one(path):
    try:
        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        h, w, _ = img_np.shape
        n = h * w
        s = img_np.sum(axis=(0, 1))
        s2 = (img_np ** 2).sum(axis=(0, 1))
        return s, s2, n
    except Exception as e:
        return None

def compute_mean_std_parallel(root_dir):
    files = []
    for subdir, _, fs in os.walk(root_dir):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg')):
                files.append(os.path.join(subdir, f))

    with Pool(processes=30) as pool:
        results = list(tqdm(pool.imap(process_one, files), total=len(files)))

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    n_pixels = 0
    for r in results:
        if r is not None:
            s, s2, n = r
            channel_sum += s
            channel_sum_sq += s2
            n_pixels += n

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)
    return mean, std


if __name__ == "__main__":
    root = "/data/wenjing/skin_dataset/ssl/ssl_data/"
    mean, std = compute_mean_std_parallel(root)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
