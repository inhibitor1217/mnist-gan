from typing import Tuple

from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam

from base.base_model import BaseModel

class DenseDiscriminator(BaseModel):
    def define_model(self, model_name: str) -> Model:
        # input
        _input = Input(shape=(784,), name=f'{model_name}_input')

        X = Dense(256, activation='relu', name=f'{model_name}_dense_1')(_input)
        X = Dense(1, activation='sigmoid', name=f'{model_name}_dense_2')(X)

        model = Model(inputs=_input, outputs=X, name=f'{model_name}')

        return model

    def build_model(self, model_name: str) -> Tuple[Model, Model]:
        model = self.define_model(model_name)

        optimizer = Adam(
            lr=self.config.model.discriminator.lr,
            beta_1=self.config.model.discriminator.beta1)
        
        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model, parallel_model