import os
from typing import Optional
from dotmap import DotMap
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
from keras import Model
from keras.callbacks import LearningRateScheduler

from base.base_trainer import BaseTrainer
from base.base_data_loader import BaseDataLoader
from utils.callback import ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, \
    TrainProgressAlertCallback, ScalarCollageTensorBoard
from utils.image import denormalize_image

class GANTrainer(BaseTrainer):
    def __init__(self, data_loader: BaseDataLoader, config: DotMap, \
                g, d, parallel_d, c, combined, combined_parallel) -> None:
        super().__init__(data_loader, config)

        

        self.init_callbacks()

    def init_callbacks(self):
        raise NotImplementedError

    @staticmethod
    def d_metric_names(model_name):
        return [f"loss/D_{model_name}", f"accuracy/D_{model_name}"]

    @staticmethod
    def g_metric_names():
        return ["loss/G"]

    def train(self):
        raise NotImplementedError

    def sample_images(self, epoch):
        raise NotImplementedError
    
    def on_batch_begin(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)