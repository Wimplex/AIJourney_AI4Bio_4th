from typing import Iterable, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset

import augmentation as aug


def apply_augment(spec: np.ndarray, augs: Iterable, feat_name: str, class_df: pd.DataFrame = None):
    for aug_ in augs:
        if isinstance(aug_, aug.SegmentsCombination):
            result = aug_(class_df[feat_name].values)
            spec = result if result is not None else spec

        elif isinstance(aug_, aug.PeakVariation) or isinstance(aug_, aug.VarianceFiltration):
            per_peak_std = class_df[feat_name].values.std()
            spec = aug_(spec, per_peak_std)

        else:
            spec = aug_(spec)
    return spec


class BioDataset(Dataset):
    def __init__(
        self, 
        data_df: pd.DataFrame, 
        augmentations: Iterable = None, 
        infer_dataset: bool = False, 
        label2id: Dict[str, int] = None,
        num_iters: int = None,
        mixup_p: float = 0.0
    ):
        self.df = data_df
        self.augs = augmentations
        self.__feats_cols = [col for col in self.df.columns if col.startswith("spec")]

        if label2id:
            self.label2id = label2id
        else:
            self.label2id = {l: i for i, l in enumerate(data_df["strain"].unique())}
        self.infer_dataset = infer_dataset
        self.num_iters = num_iters
        self.mixup_p = mixup_p

    def __onehot_encode_label(self, label: int) -> torch.Tensor:
        pholder = torch.zeros(len(self.label2id))
        pholder[label] = 1.0
        return pholder

    def __prepare_label(self, item):
        label = self.label2id[item["strain"]]
        label = self.__onehot_encode_label(label)
        return label

    def __prepare_sample(self, item):
        try:
            class_df = self.df[self.df["strain"] == item["strain"]]
        except:
            class_df = None
        specs = []
        for feat in self.__feats_cols:
            spec = item[feat]
            if not isinstance(spec, np.ndarray):
                spec = np.array(spec)
            spec = spec.squeeze()

            if self.augs and feat in self.augs.keys():
                spec = apply_augment(spec, self.augs[feat], feat, class_df).squeeze()

            spec = torch.Tensor(spec).unsqueeze(0)
            specs.append(spec)
        
        if len(specs) == 1:
            tensor = specs[0]
        else:
            tensor = torch.stack(specs, dim=0).squeeze()
        return tensor

    def __apply_mixup(self, spec, label):
        another_idx = np.random.randint(0, len(self.df))
        another_item = self.df.iloc[another_idx]
        another_spec = self.__prepare_sample(another_item)
        another_label = self.__prepare_label(another_item)
        mixup = aug.MixUp(ratio=np.random.uniform(0.6, 0.95), p=self.mixup_p)
        return mixup(spec, another_spec, label, another_label)

    def __getitem__(self, idx: int):
        idx = idx % len(self.df)
        item = self.df.iloc[idx]

        spec = self.__prepare_sample(item)

        if not self.infer_dataset:
            label = self.__prepare_label(item)
            if self.mixup_p:
                spec, label = self.__apply_mixup(spec, label)
            return spec, label
        else:
            return spec

    def __len__(self):
        if self.num_iters:
            return self.num_iters
        else:
            return len(self.df)
        