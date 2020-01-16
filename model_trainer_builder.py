from models.with_load_weights import WithLoadWeights, WithLoadOptimizerWeights
from models.classifier import Classifier

def build_model_and_trainer(config, data_loader):
    if config.model.type == 'classifier':
        model_builder = Classifier(config)
        model, parallel_model = WithLoadWeights(model_builder, model_name='classifier') \
            .build_model(model_name='classifier')
        
    elif config.model.type == 'dcgan':
        pass