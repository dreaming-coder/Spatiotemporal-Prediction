# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/9 18:23
# @author: 芜情
# @description:
import shutil
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

__all__ = ["CheckpointMonitor"]


def delete(file: Path):
    try:
        file.unlink()
    except PermissionError:
        time.sleep(1)
        delete(file)


def move(src, dest):
    try:
        shutil.move(src, dest)
    except PermissionError:
        time.sleep(1)
        move(src, dest)


class FileEventHandler(FileSystemEventHandler):

    def __init__(self, src_path: str, dest_path: str):
        self.src_path = Path(src_path)
        self.dest_path = Path(dest_path)
        if not self.dest_path.exists():
            self.dest_path.mkdir(parents=True, exist_ok=True)
        else:
            if len(list(self.dest_path.glob("checkpoint*"))) > 0:
                raise FileExistsError(
                    "you must ensure that there is no checkpoint file in the folder, this is to help you differentiate the models and datasets"
                )

    # noinspection PyShadowingNames
    def on_created(self, event):

        files = list(self.src_path.glob("checkpoint*"))
        if len(files) == 0:
            return
        cur_best_loss = min(list(map(lambda file: file.name.split("_")[2], files)))

        pre_files = list(self.dest_path.glob("checkpoint*"))
        pre_best_loss = "9999999999999999"
        if len(pre_files) > 0:
            pre_best_loss = min(list(map(lambda file: file.name.split("_")[2], pre_files)))
            if cur_best_loss < pre_best_loss:
                for pre_file in pre_files:
                    pre_file.unlink()
        for file in files:
            if cur_best_loss in file.name and cur_best_loss < pre_best_loss:
                move(file, self.dest_path.joinpath(file.name))
            else:
                delete(file)


class CheckpointMonitor(object):

    def __init__(self, src_path: str, dest_path: str):
        self.src_path = src_path
        self.dest_path = dest_path
        self.observer = Observer(timeout=300)

    def start(self):
        event_handler = FileEventHandler(self.src_path, self.dest_path)
        self.observer.schedule(event_handler, self.src_path)
        self.observer.start()

    def stop(self):
        self.observer.stop()
