from dotmap import DotMap
from typing import Generator

import numpy as np
import tensorflow as tf
from tensorpack.dataflow import BatchData, DataFlow

from base.base_data_loader import BaseDataLoader
from utils.dataflow import GeneratorToDataFlow

def load_data() -> DotMap:
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape,  test_y.shape)

    data: DotMap = DotMap({ 
        'train_x': train_x, 
        'train_y': train_y, 
        'test_x':  test_x, 
        'test_y':  test_y 
    })

    return data

def data_to_generator(x, y, shuffle: bool) -> Generator:
    while True:
        assert x.shape[0] == y.shape[0]
        data_size = x.shape[0]
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        
        for index in indices:
            yield x[index], y[index]

def data_to_dataflow(x, y, config: DotMap) -> DataFlow:
    dataflow = data_to_generator(x, y, True)
    dataflow = GeneratorToDataFlow(dataflow)
    dataflow = BatchData(dataflow, config.trainer.batch_size)
    dataflow.reset_state()

    return dataflow

class MNISTDataLoader(BaseDataLoader):
    def __init__(self, config: DotMap) -> None:
        super().__init__(config)

        data = load_data()
        
        # create train dataflow
        self.train_dataflow: DataFlow = data_to_dataflow(data.train_x, data.train_y, config)

        # create test dataflow
        self.test_dataflow:  DataFlow = data_to_dataflow(data.test_x,  data.test_y,  config)
    
    def get_train_data_generator(self) -> Generator:
        raise NotImplementedError

    def get_validation_data_generator(self) -> Generator:
        raise NotImplementedError

    def get_test_data_generator(self) -> Generator:
        raise NotImplementedError

    def get_train_data_size(self) -> int:
        raise NotImplementedError

    def get_validation_data_size(self) -> int:
        raise NotImplementedError

    def get_test_data_size(self) -> int:
        raise NotImplementedError