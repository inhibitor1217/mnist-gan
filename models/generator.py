from keras.layers import Dense, Activation, Input, Reshape, Conv2D, UpSampling2D, BatchNormalization
from keras.models import Model

from base.base_model import BaseModel


class Generator(BaseModel):
    def define_model(self, model_name):
        raise NotImplementedError

    def build_model(self, model_name):
        raise NotImplementedError