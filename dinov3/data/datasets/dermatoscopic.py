import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov3")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Dermatoscopic(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: "Dermatoscopic.Split",
        root: str,        
        extra: str,
        meta: str = "ssl_merged.csv",
        transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self._split = split
        self._extra_root = extra or os.path.join(root, "extra")        
        os.makedirs(self._extra_root, exist_ok=True)
        self.meta = meta
        self.ssl_data: str = os.path.join(root, "ssl_data")
        self._entries = self._load_entries()
        self._label_to_index = self._build_label_mapping()

    @property
    def _entries_path(self):
        return f"entries-{self._split.value.upper()}.npy"

    def _load_entries(self) -> np.ndarray:
        entries_full_path = os.path.join(self._extra_root, self._entries_path)
        if os.path.exists(entries_full_path):
            logger.info(f"Loading entries from {self._entries_path}")
            return np.load(entries_full_path, mmap_mode="r")
        else:
            return self._dump_extra()

    def _save_entries(self, entries_path, entries_array) -> None:
        entries_full_path = os.path.join(self._extra_root, entries_path)
        np.save(entries_full_path, entries_array)

    def _dump_extra(self) -> np.ndarray:
        return self._dump_entries()

    def _dump_entries(self) -> np.ndarray:
        split = self._split
        if split == Dermatoscopic.Split.TEST:
            ...
        else:
            meta_path = self.meta
            logger.info(f"Loading meta from {meta_path}")
            if os.path.exists(os.path.join(self.root, meta_path)):
                meta_full_path = os.path.join(self.root, meta_path)
            else:
                meta_full_path = os.path.join(self.ssl_data, meta_path)
            meta_df = pd.read_csv(meta_full_path)
            sample_count = len(meta_df)
            
            # build structured dtype with fixed-length Unicode fields (no object dtype)
            cols = list(meta_df.columns)

            # compute max string length per column (based on str(value)), at least 1
            col_maxlen = {}
            for col in cols:
                # convert to str and compute length (handle NaN)
                lengths = meta_df[col].astype(str).map(len)
                maxlen = int(lengths.max()) if len(lengths) > 0 else 1
                if maxlen < 1:
                    maxlen = 1
                col_maxlen[col] = maxlen

            dtype = np.dtype([(col, f"U{col_maxlen[col]}") for col in cols])

            entries = np.empty(sample_count, dtype=dtype)       
            old_percent = -1
            logger.info(f"creating entries ... ")                                
            for index, row in meta_df.iterrows():
                percent = 100 * index // sample_count
                if percent > old_percent and percent % 10 == 0:
                    logger.info(f"percent: {percent}%")
                    old_percent = percent 
                # ensure values are strings and match dtype order
                values = tuple("" if pd.isna(row[col]) else str(row[col]) for col in cols)
                entries[index] = values            
            entries_path = self._entries_path
            logger.info(f"saving entries to {entries_path}, dtype={dtype}")
            self._save_entries(entries_path, entries)
            return entries
        
    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        image_full_path = os.path.join(self.ssl_data, entry["source_dataset"], entry["image_id"] + ".jpg")
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data      

    def _build_label_mapping(self):
        labels = set()
        for e in self._entries:
            lab = str(e["diagnosis_1"]).strip()
            if lab and lab.lower() not in ("", "nan"):
                labels.add(lab)
        label_to_index = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
        logger.info(f"Built label mapping: {label_to_index}")
        return label_to_index

    def get_target(self, index: int) -> int:
        entry = self._entries[index]
        if "diagnosis_1" not in self._entries.dtype.names:
            return -1
        diag = str(entry["diagnosis_1"]).strip()
        if not diag or diag.lower() in ("", "nan"):
            return -1
        return self._label_to_index.get(diag, -1)

    def __len__(self) -> int:
        entries = self._entries
        return len(entries)