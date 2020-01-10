from typing import Tuple

from keras import Model, Input
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.layer import named

class GANCombined(BaseModel):
    def define_model(self, g: Model, d: Model, model_name: str) -> Model:
        raise NotImplementedError

    def build_model(self, g: Model, d: Model, model_name: str) -> Model:
        raise NotImplementedError