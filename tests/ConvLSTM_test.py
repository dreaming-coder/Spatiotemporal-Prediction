# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 10:12
# @author: 芜情
# @description: test for ConvLSTM

import unittest

import torch

from src.nn.ConvLSTM.ConvLSTMCell import ConvLSTMCell


class TestConvLSTM(unittest.TestCase):

    def test_ConvLSTMCell(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cell = ConvLSTMCell(in_channels=64, hidden_channels=96, size=(50, 50)).to(device)
        x = torch.ones(3, 64, 50, 50).to(device)
        h = torch.zeros(3, 96, 50, 50).to(device)
        c = torch.zeros(3, 96, 50, 50).to(device)
        hh, cc = cell(x, h, c)
        self.assertTrue(hh.shape == (3, 96, 50, 50))
        self.assertTrue(cc.shape == (3, 96, 50, 50))
        hh.sum().backward()
