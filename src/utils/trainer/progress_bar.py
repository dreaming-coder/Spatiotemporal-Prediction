# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/8 13:08
# @author: 芜情
# @description: this progress_bar function is used to generate the animation frame of progress bar

__all__ = ["progress_bar"]

dynamic_symbols = [
    "▁▂▃",
    "▂▃▄",
    "▃▄▅",
    "▄▅▆",
    "▅▆▇",
    "▆▇▆",
    "▇▆▅",
    "▆▅▄",
    "▅▄▃",
    "▄▃▂",
    "▃▂▁",
    "▂▁▂"
]


def progress_bar(cur, total):
    r"""
    :param cur: current item index
    :param total: the total count of item
    :return: the format string of progress bar
    """
    percent = cur / total
    bar_length = int(percent * 50)
    return f"|{'█' * bar_length:<50s}| {dynamic_symbols[cur % 12]} {percent:.0%}"
