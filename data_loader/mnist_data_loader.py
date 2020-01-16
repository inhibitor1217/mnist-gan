import os
import math
from datetime import datetime

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorpack.dataflow import BatchData

from base.base_data_loader import BaseDataLoader
from utils.dataflow import GeneratorToDataFlow, ProcessorDataFlow
from utils.image import normalize_image

NUM_CLASSES = 10

class MNISTDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

        # Load MNIST dataset
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

        assert x_train_raw.shape[0] == y_train_raw.shape[0]
        assert x_test_raw.shape[0] == y_test_raw.shape[0]

        self.train_data_size = x_train_raw.shape[0]
        self.valid_data_size = self.test_data_size = x_test_raw.shape[0] // 2

        # Add channel dimension
        self.x_train = np.expand_dims(x_train_raw, axis=-1)
        self.x_valid = np.expand_dims(x_test_raw[self.valid_data_size:], axis=-1)
        self.x_test  = np.expand_dims(x_test_raw[:self.test_data_size], axis=-1)

        # Format into one-hot labels
        self.y_train = np.zeros((self.train_data_size, NUM_CLASSES), dtype=float)
        self.y_train[np.arange(self.train_data_size), y_train_raw] = 1.
        self.y_valid = np.zeros((self.valid_data_size, NUM_CLASSES), dtype=float)
        self.y_valid[np.arange(self.valid_data_size), y_test_raw[self.valid_data_size:]] = 1.
        self.y_test  = np.zeros((self.test_data_size, NUM_CLASSES), dtype=float)
        self.y_test [np.arange(self.test_data_size), y_test_raw[:self.test_data_size]] = 1.

        self.train_batch_size = config.trainer.batch_size
        self.valid_batch_size = config.trainer.batch_size
        self.test_batch_size  = config.trainer.batch_size

        # Adjust data size by batch_size
        self.train_data_size = math.ceil(self.train_data_size / self.train_batch_size)
        self.valid_data_size = math.ceil(self.valid_data_size / self.valid_batch_size)
        self.test_data_size  = math.ceil(self.test_data_size  / self.test_batch_size)

    def _data_to_generator(self, x, y, shuffle):
        assert x.shape[0] == y.shape[0]       
        
        seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        np.random.seed(seed)

        indices = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        for index in indices:
            yield x[index], y[index]

    def _data_to_dataflow(self, x, y, shuffle, batch_size=1):
        def preprocess(data):
            x, y = data
            return normalize_image(x), y

        dataflow = self._data_to_generator(x, y, shuffle)
        dataflow = GeneratorToDataFlow(dataflow)
        dataflow = ProcessorDataFlow(dataflow, preprocess)
        dataflow = BatchData(dataflow, batch_size, remainder=True)
        dataflow.reset_state()
        return dataflow

    def get_train_data_generator(self):
        return self._data_to_dataflow(self.x_train, self.y_train, 
                    shuffle=True,  batch_size=self.train_batch_size).get_data()

    def get_validation_data_generator(self):
        return self._data_to_dataflow(self.x_valid, self.y_valid, 
                    shuffle=False, batch_size=self.valid_batch_size).get_data()

    def get_test_data_generator(self):
        return self._data_to_dataflow(self.x_test , self.y_test, 
                    shuffle=False, batch_size=self.test_batch_size ).get_data()

    def get_train_data_size(self):
        return self.train_data_size

    def get_validation_data_size(self):
        return self.valid_data_size

    def get_test_data_size(self):
        return self.test_data_size