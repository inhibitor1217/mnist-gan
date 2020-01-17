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

        self.g = g
        self.serial_d = d
        self.d = parallel_d
        self.c = c
        self.serial_combined = combined
        self.combined = combined_parallel

        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self):
        # if horovod used, only worker 0 saves checkpoints
        is_master = True
        is_local_master = True
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            is_master = hvd.rank() == 0
            is_local_master = hvd.local_rank() == 0

        # horovod callbacks
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            self.model_callbacks["combined"].append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            self.model_callbacks["combined"].append(hvd.callbacks.MetricAverageCallback())
            self.model_callbacks["combined"].append(
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

        if is_local_master:
            # model saver
            self.model_callbacks["serial_combined"].append(
                ModelCheckpointWithKeepFreq(
                    filepath=os.path.join(self.config.exp.checkpoints_dir, "{epoch:04d}-combined.hdf5"),
                    keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                    save_checkpoint_freq=self.config.trainer.save_checkpoint_freq,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=1))

            # save optimizer weights
            for model_name in ['combined', 'd']:
                self.model_callbacks[model_name].append(OptimizerSaver(self.config, model_name))
        if is_master:
            # save individual models
            for model_name in ['g', 'd']:
                self.model_callbacks[model_name].append(
                    ModelSaver(
                        checkpoint_dir=self.config.exp.checkpoints_dir,
                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                        model_name=model_name,
                        num_epochs=self.config.trainer.num_epochs,
                        verbose=1))

            # send notification to telegram channel on train start and end
            self.model_callbacks["combined"].append(TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                                                               total_epochs=self.config.trainer.num_epochs))

            # tensorboard callback
            self.model_callbacks["combined"].append(
                ScalarCollageTensorBoard(log_dir=self.config.exp.tensorboard_dir,
                                         batch_size=self.config.trainer.batch_size,
                                         write_images=True))

        # initialize callbacks by setting model and params
        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.data_loader.get_train_step_size()
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")

            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    "batch_size": self.config.trainer.batch_size,
                    "epochs": epochs,
                    "steps": steps_per_epoch,
                    "samples": self.data_loader.get_train_data_size(),
                    "verbose": True,
                    "do_validation": False,
                    "model_name": model_name,
                })

    def metric_string(self, metric_name, metric_value):
        if 'accuracy' in metric_name:
            return f"{metric_name}={metric_value*100:.1f}%"
        elif 'loss' in metric_name:
            return f"{metric_name}={metric_value:.4f}"
        else:
            return f"{metric_name}={metric_value}"

    def train(self):
        
        epochs = self.config.trainer.num_epochs
        start_time = datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})
            epoch_logs = {
                'loss/d_real_train': 0,
                'loss/d_fake_train': 0,
                'loss/g_total_train': 0,
                'loss/g_adversarial_train': 0,
                'loss/g_classifier_train': 0,
                'accuracy/d_real_train': 0,
                'accuracy/d_fake_train': 0
            }
            train_data = self.data_loader.get_train_data_generator()
            for idx, (x, y) in enumerate(train_data):
                step = idx + 1
                assert x.shape[0] == y.shape[0]
                batch_size = x.shape[0]

                batch_logs = {'batch': step, 'size': batch_size}
                self.on_batch_begin(step, batch_logs)

                label   = np.resize(y, (batch_size, 1, 1, 10))
                noise   = np.random.normal(0, 1, (batch_size, 1, 1, 54))
                g_input = np.concatenate([label, noise], axis=-1)

                fake = self.g.predict(g_input)

                real_prediction       = np.ones ((batch_size, 1))
                fake_prediction       = np.zeros((batch_size, 1))

                [d_loss_real, d_accuracy_real] = self.d.train_on_batch(x,    real_prediction)    # Train on real images
                [d_loss_fake, d_accuracy_fake] = self.d.train_on_batch(fake, fake_prediction)    # Train on fake images

                [g_loss_total, g_loss_adversarial, g_loss_classifier] = self.combined.train_on_batch([label, noise], [real_prediction, y])

                metric_logs = {
                    'loss/d_real_train': d_loss_real,
                    'loss/d_fake_train': d_loss_fake,
                    'loss/g_total_train': g_loss_total,
                    'loss/g_adversarial_train': g_loss_adversarial,
                    'loss/g_classifier_train': g_loss_classifier,
                    'accuracy/d_real_train': d_accuracy_real,
                    'accuracy/d_fake_train': d_accuracy_fake
                }

                batch_logs.update(metric_logs)

                # Update epoch log accordingly
                for metric_name in metric_logs.keys():
                    if metric_name in epoch_logs:
                        epoch_logs[metric_name] += metric_logs[metric_name] * batch_size # weighted average (batch size could be different!)

                iter_str   = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{self.data_loader.get_train_step_size()}]"
                metric_str = ', '.join([self.metric_string(name, value) for name, value in metric_logs.items()])
                time_str   = f"time: {datetime.now() - start_time}"
                print(', '.join([iter_str, metric_str, time_str]), flush=True)

                self.on_batch_end(step, batch_logs)

            # sum to average
            for k in epoch_logs:
                epoch_logs[k] /= self.data_loader.get_train_data_size()
            epoch_logs = dict(epoch_logs)

            # TODO: perform prediction using sample_images

            self.on_epoch_end(epoch, epoch_logs)
        self.on_train_end()

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