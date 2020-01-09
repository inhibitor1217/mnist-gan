from dotmap import DotMap
from typing import Tuple
from tensorflow.keras import Model

from base.base_data_loader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer

def build_model_and_trainer(config: DotMap, data_loader: BaseDataLoader) -> Tuple[Model, BaseTrainer]:
    pass