import os
import sys
PROJECT_ROOT = "/home/jiangwenjing/hd/rose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rose.data.datasets import ImageNet
from rose.data.datasets.dermoscopy import Dermoscopy
from rose.data.datasets import HAM10K, BCN20K, HIBA, MNDICHD

import numpy as np
"""
print(ImageNet.Split)
for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/sdu/haodi/imagenet1k-256", extra="/sdu/haodi/imagenet1k-256/extra")
    dataset.dump_extra()"""
dataset_class_list = [cls for cls in Dermoscopy.__subclasses__()]

for dataset in dataset_class_list:
    print(dataset.__name__)
    ds = dataset(split=dataset.Split.TRAIN, root="/data/wenjing/skin_dataset/ssl", extra="/data/wenjing/skin_dataset/ssl/extra")
    print(ds._build_label_mapping())
"""for split in MNDICHD.Split:
    dataset = MNDICHD(split=split, root="/data/wenjing/skin_dataset/ssl", extra="/data/wenjing/skin_dataset/ssl/extra")
    dataset._load_entries()
    #print(split)
    print(dataset._build_label_mapping())
    #print(dataset.get_target(23))"""
"""
a = np.load("/data/wenjing/skin_dataset/ssl/extra/mn-entries-TRAIN.npy", mmap_mode="r")
b = np.load("/data/wenjing/skin_dataset/ssl/extra/mn-entries-VAL.npy", mmap_mode="r")
c = np.load("/data/wenjing/skin_dataset/ssl/extra/mn-entries-TEST.npy", mmap_mode="r")
print(a.shape)
print(b.shape)
print(c.shape)
"""