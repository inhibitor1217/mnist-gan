from typing import Tuple

from keras import Model, Input
from keras.optimizers import Adam

from base.base_model import BaseModel
from utils.layer import named

class GANCombined(BaseModel):
    def define_model(self, g: Model, d: Model, model_name: str) -> Model:
        d.trainable = False

        input_noise = Input(shape=(64,), name='input_noise')

        # Fake image generated by generator
        fake_image  = g(input_noise)

        # Predictions
        logit       = d(fake_image)

        # loss = log(1 - D(G(z)))
        return Model(inputs=input_noise, outputs=logit, name=model_name)

    def build_model(self, g: Model, d: Model, model_name: str) -> Model:
        combined = self.define_model(g, d, model_name)
        optimizer = Adam(self.config.model.generator.lr)
        parallel_combined = self.multi_gpu_model(combined)
        parallel_combined.compile(optimizer=optimizer, loss='binary_crossentropy')

        return combined, parallel_combined