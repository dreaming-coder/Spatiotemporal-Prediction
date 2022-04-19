# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 14:31
# @author: 芜情
# @description:
import unittest

from src.utils.data import MovingMNISTDataset
from src.utils.data.KTHDataset import KTHDataset
from src.utils.data.TaxiBJDataset import TaxiBJDataset


class TestDataset(unittest.TestCase):

    def test_MovingMNISTDataset(self):
        dataset = MovingMNISTDataset("train")
        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][0].max())

    def test_KTHDataset(self):
        dataset = KTHDataset("test")
        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][0].max())

    def test_TaxiBJDataset(self):
        dataset = TaxiBJDataset("validation")
        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][0].max())
        print(dataset[0][0].min())