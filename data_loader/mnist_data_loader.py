from dotmap import DotMap
from typing import Generator

from base.base_data_loader import BaseDataLoader

class MNISTDataLoader(BaseDataLoader):
    def __init__(self, config: DotMap) -> None:
        super().__init__(config)


    def get_train_data_generator(self) -> Generator:
        raise NotImplementedError

    def get_test_data_generator(self) -> Generator:
        raise NotImplementedError

    def get_train_data_size(self) -> int:
        raise NotImplementedError

    def get_test_data_size(self) -> int:
        raise NotImplementedError