# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/13 9:00
# @author: 芜情
# @description:
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.data import TaxiBJDataset


# dram a sequence of prediction among one sample
def plot_Moving_MNIST_Prediction():
    data = torch.load("../results/MovingMNIST/TrajGRU/prediction/pred_00101.pth")[0]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 11:02d}.png", data[seq, 0], cmap="gray", vmin=0, vmax=1.0)


def plot_KTH_Prediction():
    data = torch.load("../results/KTH/TrajGRU/prediction/pred_00001.pth")[0]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 11:02d}.png", data[seq, 0], cmap="gray", vmin=0, vmax=1.0)


def plot_TaxiBJ_Prediction():
    data = torch.load("../results/TaxiBJ/TrajGRU/prediction/pred_00021.pth")[0]
    data_label = TaxiBJDataset("test")[0][1]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 5:02d}.svg", data[seq, 0] - data_label[seq, 0], vmin=0, vmax=1.0)


if __name__ == '__main__':
    plot_TaxiBJ_Prediction()