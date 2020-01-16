import os
from collections import defaultdict
from datetime import datetime

from keras.callbacks import LearningRateScheduler

from base.base_trainer import BaseTrainer
from utils.callback import ScalarCollageTensorBoard, ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, \
    TrainProgressAlertCallback

class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, parallel_model, data_loader, config):
        super().__init__(data_loader, config)
        self.serial_model = model
        self.model = parallel_model

        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self):
        if self.config.trainer.use_lr_decay:
            # linear decay from the half of max_epochs
            def lr_scheduler(lr, epoch, max_epochs):
                return min(lr, 2 * lr * (1 - epoch / max_epochs))

            self.model_callbacks['model'].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch,
                                                                          self.config.trainer.num_epochs)))

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

            self.model_callbacks["model"].append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            self.model_callbacks["model"].append(hvd.callbacks.MetricAverageCallback())
            self.model_callbacks["model"].append(
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

        if is_local_master:
            # model saver
            self.model_callbacks["serial_model"].append(
                ModelCheckpointWithKeepFreq(
                    filepath=os.path.join(self.config.exp.checkpoints_dir, "{epoch:04d}-combined.hdf5"),
                    keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                    save_checkpoint_freq=self.config.trainer.save_checkpoint_freq,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=1))

            # save optimizer weights
            for model_name in ['model']:
                self.model_callbacks[model_name].append(OptimizerSaver(self.config, model_name))
        if is_master:
            # save individual models
            for model_name in ['model']:
                self.model_callbacks[model_name].append(
                    ModelSaver(
                        checkpoint_dir=self.config.exp.checkpoints_dir,
                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                        model_name=model_name,
                        num_epochs=self.config.trainer.num_epochs,
                        verbose=1))

            # send notification to telegram channel on train start and end
            self.model_callbacks["model"].append(TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                                                               total_epochs=self.config.trainer.num_epochs))

            # tensorboard callback
            self.model_callbacks["model"].append(
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

    def train(self):
        def metric_string(metric_name, metric_value):
            if 'accuracy' in metric_name:
                return f"{metric_name}={metric_value*100:.1f}%"
            elif 'loss' in metric_name:
                return f"{metric_name}={metric_value:.4f}"
            else:
                return f"{metric_name}={metric_value}"
   
        epochs = self.config.trainer.num_epochs
        start_time = datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})
            epoch_logs = { 'loss/train': 0., 'accuracy/train': 0. }

            train_data = self.data_loader.get_train_data_generator()
            
            for idx, (x, y) in enumerate(train_data):
                step = idx + 1

                assert x.shape[0] == y.shape[0]
                batch_size = x.shape[0]

                batch_logs = {'batch': step, 'size': batch_size}
                self.on_batch_begin(step, batch_logs)

                [loss, accuracy] = self.model.train_on_batch(x, y)

                metric_logs = {
                    'loss/train': loss,
                    'accuracy/train': accuracy
                }

                batch_logs.update(metric_logs)

                # Update epoch log accordingly
                for metric_name in metric_logs.keys():
                    if metric_name in epoch_logs:
                        epoch_logs[metric_name] += metric_logs[metric_name] * batch_size # weighted average (batch size could be different!)

                iter_str   = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{self.data_loader.get_train_step_size()}]"
                metric_str = ', '.join([metric_string(name, value) for name, value in metric_logs.items()])
                time_str   = f"time: {datetime.now() - start_time}"
                print(', '.join([iter_str, metric_str, time_str]), flush=True)

                self.on_batch_end(step, batch_logs)

            # sum to average
            for k in epoch_logs:
                epoch_logs[k] /= self.data_loader.get_train_data_size()
            epoch_logs = dict(epoch_logs)

            if (epoch + 1) % self.config.trainer.predict_freq == 0:
                self.predict_validation(epoch + 1)

            self.on_epoch_end(epoch, epoch_logs)
        
        self.predict_test()
        self.on_train_end()

    def predict_validation(self, epoch):
        print(f"Prediction for validation set at epoch {epoch}", flush=True)

        valid_data = self.data_loader.get_validation_data_generator()
        valid_size = self.data_loader.get_validation_data_size()

    def predict_test(self):
        print(f"Prediction for test set")

        test_data = self.data_loader.get_test_data_generator()
        test_size = self.data_loader.get_test_data_size()

    def on_batch_begin(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)