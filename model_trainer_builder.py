import os

from models.with_load_weights import WithLoadWeights, WithLoadOptimizerWeights
from models.classifier import Classifier
from models.discriminator import Discriminator
from models.generator import Generator
from models.gan_combined import GANCombined
from trainers.classifier_trainer import ClassifierTrainer
from trainers.gan_trainer import GANTrainer

def build_model_and_trainer(config, data_loader):
    if config.model.type == 'classifier':
        model_builder = Classifier(config)
        model, parallel_model = WithLoadWeights(model_builder, model_name='classifier') \
            .build_model(model_name='classifier')
        trainer = ClassifierTrainer(model, parallel_model, data_loader, config)

        return model, trainer

    elif config.model.type == 'dcgan':
        g_model_builder = Generator(config)
        d_model_builder = Discriminator(config)
        # c_model_builder = Classifier(config)

        g = g_model_builder.define_model('generator')
        d, parallel_d = d_model_builder.build_model('discriminator')
        # c, _ = c_model_builder.build_model('classifier')

        # Load weights to classifier
        # checkpoint_path = './experiments/classifier_mnist/checkpoints/0050-classifier.hdf5'
        # if os.path.exists(checkpoint_path):
        #     c.load_weights(checkpoint_path)

        combined_model_builder = GANCombined(config)

        combined, parallel_combined = WithLoadWeights(combined_model_builder, model_name='combined') \
            .build_model(g=g, d=d, model_name='combined')
            # .build_model(g=g, d=d, c=c, model_name='combined')
        # trainer = GANTrainer(data_loader, config, g, d, parallel_d, c, combined, parallel_combined)
        trainer = GANTrainer(data_loader, config, g, d, parallel_d, combined, parallel_combined)

        return combined, trainer
