from keras.layers import Dense, Input, Flatten, Conv2D
from keras.models import Model
from keras.optimizers import SGD

from base.base_model import BaseModel


class Discriminator(BaseModel):
    def define_model(self, model_name):
        raise NotImplementedError

    def build_model(self, model_name):
        raise NotImplementedError