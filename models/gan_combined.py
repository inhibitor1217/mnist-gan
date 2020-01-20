from keras import Model, Input
from keras.layers import Concatenate
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.layer import named

class GANCombined(BaseModel):
    def define_model(self, g, d, c, model_name):
        c.trainable = False
        d.trainable = False

        # _input_label = Input(shape=(1, 1, 10), name=f'{model_name}_input_label')
        # _input_noise = Input(shape=(1, 1, 54), name=f'{model_name}_input_noise')

        # _input = Concatenate(axis=-1, name=f'{model_name}_input')([_input_label, _input_noise])
        _input = Input(shape=(1, 1, 64), name=f'{model_name}_input')

        fake = g(_input)

        fake_prediction = d(fake)
        # fake_classification = c(fake)

        #model = Model(inputs=[_input_label, _input_noise], outputs=[fake_prediction, fake_classification, fake], name=model_name)
        model = Model(inputs=_input, outputs=fake_prediction, name=model_name)

        return model

    def build_model(self, g, d, c, model_name):
        model = self.define_model(g, d, c, model_name)
        optimizer = Adam(self.config.model.generator.lr, beta_1=self.config.model.generator.beta1,
                        clipvalue=self.config.model.generator.clipvalue,
                        clipnorm=self.config.model.generator.clipnorm)
        parallel_model = self.multi_gpu_model(model)
        # parallel_model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'binary_crossentropy', 'mae'],
        #             loss_weights=[1, self.config.model.generator.weight_classifier, self.config.model.generator.weight_l1])
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model, parallel_model