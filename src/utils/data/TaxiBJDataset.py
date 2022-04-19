# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/19 13:57
# @author: 芜情
# @description:
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["TaxiBJDataset"]

from resources.setup import TaxiBJ


class TaxiBJDataset(Dataset):
    def __init__(self, dataset: str):
        r"""
        数据集 0-1 归一化，按照测试集最小值 0， 最大值 1292
        """
        if dataset == "train":
            self.dataset_dir = Path(TaxiBJ + r"\train")
        elif dataset == "validation":
            self.dataset_dir = Path(TaxiBJ + r"\validation")
        elif dataset == "test":
            self.dataset_dir = Path(TaxiBJ + r"\test")
        else:
            raise FileNotFoundError(f"\nthe dataset {dataset} in Moving MNIST doesn't exist.\n")

        self.__len = len(list(self.dataset_dir.glob("*.npy")))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        example_path = str(self.dataset_dir.joinpath(f"example{index + 1:06d}.npy"))
        sequence = np.load(example_path)
        sequence = torch.from_numpy(sequence).float() / 1292.0

        inputs, labels = torch.split(sequence, 4)

        return inputs, labels

    def __len__(self):
        return self.__len
