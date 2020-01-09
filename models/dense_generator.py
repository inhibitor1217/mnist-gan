from typing import Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from base.base_model import BaseModel

class DenseGenerator(BaseModel):
    def define_model(self, model_name: str) -> Model:
        # input is a noise with shape (64,)
        _input = Input(shape=(None, 64), name=f'{model_name}_input')

        X = Dense(256, activation='relu', name=f'{model_name}_dense_1')(_input)
        X = Dense(784, activation='sigmoid', name=f'{model_name}_dense_2')(X)

        model = Model(inputs=_input, outputs=X, name=model_name)

        return model

    def build_model(self, **kargs) -> Model:
        raise NotImplementedError