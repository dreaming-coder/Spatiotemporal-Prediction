# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/13 9:00
# @author: 芜情
# @description:
import matplotlib.pyplot as plt
import numpy as np
import torch


# dram a sequence of prediction among one sample
def plot_Moving_MNIST_Prediction():
    data = torch.load("../results/MovingMNIST/ConvLSTM/prediction/pred_00101.pth")[0]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 11:02d}.png", data[seq, 0], cmap="gray")


def plot_KTH_Prediction():
    data = torch.load("../results/KTH/ConvLSTM/prediction/pred_00001.pth")[0]
    for seq in range(data.shape[0]):
        plt.imsave(f"{seq + 11:02d}.png", data[seq, 0], cmap="gray")


def plot_TaxiBJ_Prediction():
    ...


if __name__ == '__main__':
    plot_Moving_MNIST_Prediction()
