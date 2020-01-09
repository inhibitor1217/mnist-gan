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
    generator_builder = Generator(config)
    discriminator_builder = Discriminator(config)

    g = generator_builder.define_model(model_name='g')
    d, d_parallel = WithLoadOptimizerWeights(discriminator_builder, model_name='d').build_model(model_name='d')
    combined, combined_parallel = WithLoadWeights(GANCombined(config), model_name='combined') \
        .build_model(g=g, d=d, model_name='combined')

    trainer = GANTrainer(g, d, d_parallel, combined, combined_parallel, data_loader, config)

    return combined, trainer