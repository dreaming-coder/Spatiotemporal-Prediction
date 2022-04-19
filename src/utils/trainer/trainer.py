# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/8 14:55
# @author: 芜情
# @description: the abstract training or testing process of model
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from nn import EnhancedModule
from utils.trainer.file_monitor import CheckpointMonitor
from utils.trainer.module_helpers import is_overridden
from utils.trainer.progress_bar import progress_bar


class Trainer(object):

    def __init__(
            self, *,
            max_epoch: int,
            device: str = None,
            to_save: str,
            seed: int = 2022
    ):
        self.max_epoch = max_epoch
        self.device = device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to_save = to_save

        self.monitor = CheckpointMonitor(src_path=".", dest_path=to_save)

        # fix the seed in order to keep the idempotence
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def fit(
            self,
            model: EnhancedModule,
            train_loader: DataLoader,
            validation_loader: DataLoader,
            ckpt_path: Optional[str] = None
    ):
        start_epoch = 1
        model.optimizer = model.configure_optimizer()
        model.lr_scheduler = model.configure_lr_scheduler(model.optimizer)
        model.to(self.device)

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            model.optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint and model.lr_scheduler is not None:
                model.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        total_per_epoch = len(train_loader)

        self.monitor.start()

        for epoch in range(start_epoch, self.max_epoch + 1):

            # training loop
            model.train()

            for batch_index, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                model.optimizer.zero_grad(set_to_none=True)

                training_outs = model.training_step(inputs, labels)
                if is_overridden("training_step_end", model):
                    training_outs = model.training_step_end(training_outs)

                if isinstance(training_outs, Tensor):
                    training_outs.backward()
                elif isinstance(training_outs, dict):
                    try:
                        training_outs["loss"].backward()
                    except KeyError:
                        sys.stderr.write("\nif the training outputs is a dictionary, it must has a key named 'loss'.\n")
                else:
                    raise TypeError(f"\nthe training_outputs [{training_outs}] is unable to backward().\n")

                # update the optimizer
                model.optimizer_step()
                # update the learning rate
                model.lr_scheduler_step()

                sys.stdout.write(f"\r\33[36mEpoch {epoch:06d} {progress_bar(batch_index + 1, total_per_epoch)}\33[0m")
                sys.stdout.flush()

            torch.cuda.empty_cache()

            # validation loop
            model.eval()
            mean_loss = 0.0

            for index, (inputs, labels) in enumerate(validation_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                validation_outs = model.validation_step(inputs, labels)
                if is_overridden("validation_step_end", model):
                    validation_outs = model.validation_step_end(validation_outs)

                if isinstance(validation_outs, Tensor):
                    mean_loss = (mean_loss * index + validation_outs.detach().cpu().item()) / (index + 1)
                elif isinstance(validation_outs, dict):
                    try:
                        mean_loss = (mean_loss * index + validation_outs["loss"].detach().cpu().item()) / (index + 1)
                    except KeyError:
                        sys.stderr.write(
                            "\nif the validation outputs is a dictionary, it must has a key named 'loss'.\n")
                else:
                    raise TypeError(f"\nthe validation_outs [{validation_outs}] is unable to compute the loss.\n")

            sys.stdout.write(
                f"\r\33[36mEpoch {epoch:06d} {progress_bar(1, 1)} loss={mean_loss:.10f}\33[0m"
            )
            sys.stdout.flush()

            print()  # just wrap around for the log information of each epoch shows in different lines

            # save epoch results
            states_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": model.optimizer.state_dict()
            }
            if model.lr_scheduler is not None:
                states_dict["lr_scheduler"] = model.lr_scheduler.state_dict()

            torch.save(obj=states_dict, f=f"checkpoint_{epoch:06d}_{mean_loss:.10f}_temp.pth")

        self.monitor.stop()

        # clear the GPU memery
        torch.cuda.empty_cache()

    def predict(
            self,
            model: EnhancedModule,
            test_loader: DataLoader,
            ckpt_path: str
    ):
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
        except FileNotFoundError:
            sys.stderr.write(f"\nthe file {ckpt_path} does not exist.\n")
        except IOError:
            sys.stderr.write(
                f"\nthere is some wrong when loading parameters, please ensure {ckpt_path} is a right path.\n")

        model.eval()
        model.to(self.device)

        assert test_loader.batch_size == 1

        # test loop
        total_per_epoch = len(test_loader)
        prediction_home = Path(self.to_save).joinpath("prediction")
        if not prediction_home.exists():
            prediction_home.mkdir()
        for batch_index, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model.predict_step(inputs, labels)
            torch.save(outputs.detach().cpu(), f=prediction_home.joinpath(f"pred_{batch_index + 1:05d}.pth"))

            sys.stdout.write(f"\r\33[94m正在处理 {progress_bar(batch_index + 1, total_per_epoch)}\33[0m")
            sys.stdout.flush()

        torch.cuda.empty_cache()

        sys.stdout.write(f"\r\33[94m处理完毕 {progress_bar(total_per_epoch, total_per_epoch)}\33[0m")
