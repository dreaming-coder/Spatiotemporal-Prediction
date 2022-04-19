# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/9 9:46
# @author: 芜情
# @description:
from typing import Union, Dict, Any, List

import torch
# noinspection PyProtectedMember
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
Scheduler = Union[_LRScheduler, ReduceLROnPlateau]
