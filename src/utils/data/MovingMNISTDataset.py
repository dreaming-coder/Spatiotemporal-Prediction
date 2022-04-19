# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 13:46
# @author: 芜情
# @description: Moving Mnist dataset
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from resources.setup import MovingMNIST

__all__ = ["MovingMNISTDataset"]


class MovingMNISTDataset(Dataset):

    def __init__(self, dataset: str):
        r"""
        数据集像素值已被归一化到[0,1]范围内
        """
        if dataset == "train":
            self.dataset_dir = Path(MovingMNIST + r"\train")
        elif dataset == "validation":
            self.dataset_dir = Path(MovingMNIST + r"\validation")
        elif dataset == "test":
            self.dataset_dir = Path(MovingMNIST + r"\test")
        else:
            raise FileNotFoundError(f"\nthe dataset {dataset} in Moving MNIST doesn't exist.\n")

        self.__len = len(list(self.dataset_dir.glob("*.npz")))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        example_path = str(self.dataset_dir.joinpath(f"example{index + 1:06d}.npz"))
        dataset = np.load(example_path)

        inputs = dataset["inputs"]
        labels = dataset["labels"]

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        return inputs, labels

    def __len__(self):
        return self.__len
