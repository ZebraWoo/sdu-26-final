import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union
import dinov3.distributed as distributed
import numpy as np
import pandas as pd

from .dermoscopy import Dermoscopy
from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class HAM10K(Dermoscopy):
    Split = Union[_Split]

    def __init__(
        self,
        split: "HAM10K.Split",
        root: str,        
        extra: str,
        meta: str = "HAM.csv",
        transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        super().__init__(
            split=split,
            root=root,
            extra=extra,
            meta=meta,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform,
        )
        
    @property
    def _entries_path(self):
        return f"ham-entries-{self._split.value.upper()}.npy"
    
    def _load_entries(self) -> np.ndarray:
        entries_full_path = os.path.join(self._extra_root, self._entries_path)
        if os.path.exists(entries_full_path):
            logger.info(f"Loading entries from {self._entries_path}")
            return np.load(entries_full_path, mmap_mode="r")
        else:
            return self._dump_extra()

    def _dump_extra(self) -> np.ndarray:
        return self._dump_entries()

    def _dump_entries(self) -> np.ndarray:
        split = self._split
        meta_path = self.meta
        logger.info(f"Loading meta from {meta_path}")

        if os.path.exists(os.path.join(self.root, meta_path)):
            meta_full_path = os.path.join(self.root, meta_path)
        else:
            meta_full_path = os.path.join(self.ssl_data, meta_path)

        meta_df = pd.read_csv(meta_full_path)
        if "isic_id" in meta_df.columns:
            meta_df = meta_df.rename(columns={"isic_id": "image_id"})
        logger.info(f"Loaded {len(meta_df)} samples from {meta_full_path}")

        # 设置随机种子，保证划分可复现
        np.random.seed(self.seed)
        indices = np.arange(len(meta_df))
        np.random.shuffle(indices)
        n = len(indices)
        train_end = int(0.7 * n)
        val_end = int(0.9 * n)
        # 如果是训练集或验证集，则先划分
        if split == HAM10K.Split.TRAIN:
            selected_idx = indices[:train_end]
            logger.info(f"Using {len(selected_idx)} samples for TRAIN split")
        elif split == HAM10K.Split.VAL:
            selected_idx = indices[train_end:val_end]
            logger.info(f"Using {len(selected_idx)} samples for VAL split")
        elif split == HAM10K.Split.TEST:  # ✅ 新增 test 集划分
            selected_idx = indices[val_end:]
            logger.info(f"Using {len(selected_idx)} samples for TEST split")
        else:
            raise ValueError(f"Unknown split: {split}")
        selected_df = meta_df.iloc[selected_idx].reset_index(drop=True)
        sample_count = len(selected_df)
        cols = list(selected_df.columns)

        # 计算每列字符串的最大长度
        col_maxlen = {}
        for col in cols:
            lengths = selected_df[col].astype(str).map(len)
            maxlen = int(lengths.max()) if len(lengths) > 0 else 1
            if maxlen < 1:
                maxlen = 1
            col_maxlen[col] = maxlen

        dtype = np.dtype([(col, f"U{col_maxlen[col]}") for col in cols])
        entries = np.empty(sample_count, dtype=dtype)

        logger.info(f"Creating structured entries with dtype={dtype}")
        old_percent = -1
        for index, row in selected_df.iterrows():
            percent = 100 * index // sample_count
            if percent > old_percent and percent % 10 == 0:
                logger.info(f"Progress: {percent}%")
                old_percent = percent
            values = tuple("" if pd.isna(row[col]) else str(row[col]) for col in cols)
            entries[index] = values

        entries_path = self._entries_path
        logger.info(f"Saving entries to {entries_path}")
        self._save_entries(entries_path, entries)
        return entries
    
    def get_image_data(self, index: int) -> bytes:
        entry = self._get_entries[index]
        image_full_path = os.path.join(self.ssl_data, "HAM", entry["image_id"] + ".jpg")
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data
    
    def _build_label_mapping(self):
        labels = set()
        for e in self._get_entries:
            lab = str(e["dx"]).strip()
            if lab and lab.lower() not in ("", "nan"):
                labels.add(lab)
        label_to_index = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
        if distributed.is_main_process():
            logger.info(f"built label mapping: {label_to_index}")
        return label_to_index

    def get_target(self, index: int) -> int:
        entry = self._get_entries[index]
        if "dx" not in self._entries.dtype.names:
            return -1
        diag = str(entry["dx"]).strip()
        if not diag or diag.lower() in ("", "nan"):
            return -1
        if self._label_to_index is None:
            self._label_to_index = self._build_label_mapping()
        return self._label_to_index.get(diag, -1)
    
    @property
    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def __len__(self) -> int:
        return len(self._get_entries)