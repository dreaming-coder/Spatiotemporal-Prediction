# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/11 16:21
# @author: 芜情
# @description: the trick for limit GPU memery

from torch import Tensor

__all__ = ["reshape_patch", 'reshape_patch_back']


def reshape_patch(img_tensor: Tensor, patch_size: int) -> Tensor:
    if patch_size == 1:
        return img_tensor

    assert 5 == img_tensor.ndim
    batch, seq, channel, height, width = img_tensor.shape

    if height % patch_size or width % patch_size:
        raise ValueError("\npatch must divide origin tensor to integers.\n")

    a = img_tensor.permute(0, 1, 3, 4, 2)
    b = a.reshape(batch, seq,
                  height // patch_size, patch_size,
                  width // patch_size, patch_size,
                  channel)
    c = b.permute(0, 1, 2, 4, 3, 5, 6)
    d = c.reshape(batch, seq,
                  height // patch_size,
                  width // patch_size,
                  patch_size * patch_size * channel)
    patch_tensor = d.permute(0, 1, 4, 2, 3)

    return patch_tensor


def reshape_patch_back(patch_tensor: Tensor, patch_size: int) -> Tensor:
    if patch_size == 1:
        return patch_tensor

    assert 5 == patch_tensor.ndim
    batch, seq, channel, height, width = patch_tensor.shape
    a = patch_tensor.permute(0, 1, 3, 4, 2)
    img_channel = channel // (patch_size * patch_size)

    b = a.reshape(batch, seq,
                  height, width,
                  patch_size, patch_size,
                  img_channel)
    c = b.permute(0, 1, 2, 4, 3, 5, 6)
    d = c.reshape(batch, seq,
                  height * patch_size,
                  width * patch_size,
                  img_channel)

    img_tensor = d.permute(0, 1, 4, 2, 3)

    return img_tensor
