# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/29 13:09
# @author: 芜情
# @description:

import unittest

import torch

from src.nn.TrajGRU import TrajGRU_TaxiBJ
from src.nn.TrajGRU import TrajGRU_KTH
from src.nn.TrajGRU import TrajGRU_MovingMNIST
from src.nn.TrajGRU.TrajGRUCell import TrajGRUCell


class TestTrajGRU(unittest.TestCase):

    def test_TrajGRUCell(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cell = TrajGRUCell(in_channels=64, hidden_channels=96).to(device)
        x = torch.ones(3, 64, 32, 32).to(device)
        h = torch.zeros(3, 96, 32, 32).to(device)
        hh, hh = cell(x, h)
        self.assertTrue(hh.shape == (3, 96, 32, 32))
        hh.sum().backward()

    def test_TrajGRU_MovingMNIST(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = TrajGRU_MovingMNIST().to(device)
        x = torch.ones((3, 10, 1, 64, 64)).to(device)
        r = net(x)
        print(r.shape)

    def test_TrajGRU_KTH(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = TrajGRU_KTH().to(device)
        x = torch.ones((3, 10, 1, 128, 128)).to(device)
        r = net(x)
        print(r.shape)

    def test_TrajGRU_TaxiBJ(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = TrajGRU_TaxiBJ().to(device)
        x = torch.ones((3, 4, 1, 32, 32)).to(device)
        r = net(x)
        print(r.shape)