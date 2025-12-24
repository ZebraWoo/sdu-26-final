import os
import csv


root_dir = "/sdu/haodi/imagenet1k-256"
full_mapping = os.path.join(root_dir, "imagenet_synsets.txt")
target_dir = os.path.join(root_dir, "train")

with os.scandir(target_dir) as sd:
    target_cls = [sub_dir.name for sub_dir in sd]
cls_mapping = []
with open(full_mapping, "r") as f:
    for cls in f:
        item = cls.strip().split(" ")[0]
        if item in target_cls:
            cls_mapping.append(cls)

with open(os.path.join(root_dir, "cls_mapping.csv"), "w") as d:
    writer = csv.writer(d)
    for cls in cls_mapping:
        class_index, class_name = cls.split(" ", 1)[0], cls.split(" ", 1)[1]
        writer.writerow([class_index, class_name])










