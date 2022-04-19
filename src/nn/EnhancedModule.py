# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/9 9:25
# @author: 芜情
# @description: the enhancement of torch.nn.Module
import abc
from abc import abstractmethod
from typing import Optional

from torch import nn, Tensor
from torch.optim import Optimizer

from utils.types import STEP_OUTPUT, Scheduler

__all__ = ["EnhancedModule"]


# noinspection PyIncorrectDocstring
class EnhancedModule(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super(EnhancedModule, self).__init__()
        self.optimizer = None
        self.lr_scheduler = None

    @abstractmethod
    def configure_optimizer(self) -> Optimizer: ...

    def configure_lr_scheduler(self, optimizer: Optimizer) -> Scheduler: ...

    @abstractmethod
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        r"""
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            inputs: the tensor input into the model
            labels: the truth corresponds to the inputs
        Return:
            Any of.
            - torch.Tensor: the loss Tensor
            - dict: A dictionary. Can include any keys, but must include the key 'loss'
            - None: Training will skip to the next batch.
        Example:
            def training_step(self, inputs, labels):
                out = self.encoder(inputs)
                loss = self.loss(out, labels)
                return loss
        """

    # def training_epoch_end(self, step_output: EPOCH_OUTPUT) -> None:
    #     r"""
    #     Called at the end of the training epoch with the outputs of all training steps. Use this in case you
    #     need to do something with all the outputs returned by training_step.
    #
    #     Args:
    #         outputs: List of outputs you defined in training_step.
    #
    #     Return:
    #         None
    #
    #     Example:
    #         train_outs = []
    #         for train_batch in train_data:
    #             out = training_step(train_batch)
    #             train_outs.append(out)
    #         training_epoch_end(train_outs)
    #
    #     Note:
    #         If this method is not overridden, this won't be called.
    #     """

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        r"""
        Use this just in case that some works haven't been completed in the training_step.

        Args:
            step_output: What you return in `training_step` for each batch part.

        Return:
            Anything
        """

    @abstractmethod
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        r"""
        see also training_step, this is used to valid the effectiveness of the model.
        """

    # def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
    #     r"""
    #     see also training_epoch_end, this is used to do some work at the end of validation step
    #     """

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        r"""
        see also training_step_end, this is for some works haven't been completed in the validation_step.
        """

    def optimizer_step(self):
        self.optimizer.step()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    @abstractmethod
    def predict_step(self, *args, **kwargs) -> Tensor:
        r"""
        Here you generate the batch outputs Tensor

        Args:
            inputs: the tensor input into the model
            labels: the truth corresponds to the inputs
        Return:
            - torch.Tensor: the loss Tensor
        Example:
            def predict_step(self, inputs, labels):
                outputs = self.encoder(inputs)
                return outputs
        """
