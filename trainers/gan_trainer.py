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
    def __init__(self, g: Model, d: Model, d_parallel: Model, combined: Model, combined_parallel: Model,
        data_loader: BaseDataLoader, config: DotMap) -> None:

        super().__init__(data_loader, config)

        self.g = g
        self.serial_d = d
        self.d = d_parallel
        self.serial_combined = combined
        self.combined = combined_parallel

        self.model_callbacks: dict = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self):
        # linear decay from the half of max_epochs
        def lr_scheduler(lr, epoch, max_epochs):
            return min(lr, 2 * lr * (1 - epoch / max_epochs))

        for model_name in ['combined', 'd']:
            self.model_callbacks[model_name].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch, self.config.trainer.num_epochs))
            )

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
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
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

    @staticmethod
    def d_metric_names(model_name):
        return [f"loss/D_{model_name}", f"accuracy/D_{model_name}"]

    @staticmethod
    def g_metric_names():
        return ["loss/G"]

    def train(self):
        train_data_generator = self.data_loader.get_train_data_generator()
        batch_size = self.config.trainer.batch_size

        steps_per_epoch = self.data_loader.get_train_data_size() // batch_size
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd
            steps_per_epoch //= hvd.size()
        assert steps_per_epoch > 0

        fake = np.zeros(shape=(self.config.trainer.batch_size,), dtype=np.float32)
        real = np.ones(shape=(self.config.trainer.batch_size,), dtype=np.float32)

        epochs = self.config.trainer.num_epochs
        start_time = datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})

            epoch_logs = defaultdict(float)

            for step in range(1, steps_per_epoch + 1):
                batch_logs = { 'batch': step, 'size': self.config.trainer.batch_size }
                self.on_batch_begin(step, batch_logs)

                noise = np.random.normal(0, 1, (self.config.trainer.batch_size, 64))
                generated_images = self.g.predict(noise)
                real_images, _ = next(train_data_generator)

                loss_fake_d = self.d.train_on_batch(generated_images, fake)
                loss_real_d = self.d.train_on_batch(real_images, real)

                loss_g = self.combined.train_on_batch(noise, real)
                
                metrics = [
                    (['g/loss'], [loss_g]),
                    (['d/loss_fake', 'd/accuracy_fake'], loss_fake_d),
                    (['d/loss_real', 'd/accuracy_real'], loss_real_d)
                ]

                for (metric_names, metric_values) in metrics:
                    for name, value in zip(metric_names, metric_values):
                        batch_logs[name] = value

                    # print
                    print_str = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{steps_per_epoch}]"
                    deliminator = ' '
                    for metric_name, metric_value in zip(metric_names, metric_values):
                        if 'acc' in metric_name:
                            metric_value = metric_value * 100
                        epoch_logs[metric_name] += metric_value
                        if 'acc' in metric_name:
                            print_str += f"{deliminator}{metric_name}={metric_value:.1f}%"
                        elif 'loss' in metric_name:
                            print_str += f"{deliminator}{metric_name}={metric_value:.4f}"
                        else:
                            print_str += f"{deliminator}{metric_name}={metric_value}"
                        if deliminator == ' ':
                            deliminator = ',\t'

                    print_str += f", time: {datetime.now() - start_time}"
                    print(print_str, flush=True)

                    for metric_name, metric_value in zip(metric_names, metric_values):
                        epoch_logs[metric_name] = metric_value

                self.on_batch_end(step, batch_logs)
        
            for k in epoch_logs:
                epoch_logs[k] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)

            self.on_epoch_end(epoch, epoch_logs)

            if (epoch + 1) % self.config.trainer.predict_freq == 0:
                self.sample_images(epoch)

        self.on_train_end()

    def sample_images(self, epoch):
        output_dir = f"{self.config.trainer.predicted_dir}/{epoch + 1}/"
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for _ in range(8):
            noise = np.random.normal(0, 1, (1, 64))
            image = self.g.predict(noise)
            image = image.reshape((28, 28))
            image = denormalize_image(image)
            images.append(image)

        for i, images in enumerate(images):
            Image.fromarray(image).save(f"{output_dir}/{i}.png")
    
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