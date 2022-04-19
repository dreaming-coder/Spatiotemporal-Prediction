# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/14 9:28
# @author: 芜情
# @description: KTH dataset
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from resources.setup import KTH


class KTHDataset(Dataset):

    def __init__(self, dataset: str):
        r"""
        数据集像素值已被归一化到[0,1]范围内
        """
        if dataset == "train":
            self.dataset_dir = Path(KTH + r"\train")
        elif dataset == "validation":
            self.dataset_dir = Path(KTH + r"\validation")
        elif dataset == "test":
            self.dataset_dir = Path(KTH + r"\test")
        else:
            raise FileNotFoundError(f"\nthe dataset {dataset} in Moving MNIST doesn't exist.\n")

        self.__len = len(list(self.dataset_dir.iterdir()))

    # noinspection PyTypeChecker
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        example_path = self.dataset_dir.joinpath(f"example{index + 1:06d}")
        sequences = []
        for file in example_path.iterdir():
            img = Image.open(file).convert('L')
            img = np.expand_dims(np.array(img), axis=0)
            sequences.append(img)

        sequences = torch.from_numpy(np.stack(sequences)) / 255.

        return sequences[:10], sequences[10:]

    def __len__(self):
        return self.__len
