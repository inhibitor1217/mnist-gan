from dotmap import DotMap
from typing import Tuple
from keras import Model

from base.base_data_loader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from models.generator import Generator
from models.discriminator import Discriminator
from models.with_load_weights import WithLoadWeights, WithLoadOptimizerWeights
from models.gan_combined import GANCombined
from trainers.gan_trainer import GANTrainer

def build_model_and_trainer(config: DotMap, data_loader: BaseDataLoader) -> Tuple[Model, BaseTrainer]:
    if config.model.type == 'classifier':
        pass
    elif config.model.type == 'dcgan':
        pass