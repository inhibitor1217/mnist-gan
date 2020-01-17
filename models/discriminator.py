from keras.models import Input, Model
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense
from keras.optimizers import Adam

from base.base_model import BaseModel


class Discriminator(BaseModel):
    def define_model(self, model_name):
        _input = Input(shape=(28, 28, 1), name=f'{model_name}_input')

        x = Conv2D(filters=4, kernel_size=3, strides=2, activation='relu', name=f'{model_name}_conv1')(_input)
        x = BatchNormalization()(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=8, kernel_size=3, strides=2, activation='relu', name=f'{model_name}_conv2')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', name=f'{model_name}_conv3')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(1, activation='tanh')(x)

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, model_name):
        model = self.define_model(model_name)

        optimizer = Adam(lr=self.config.model.discriminator.lr, beta_1=self.config.model.discriminator.beta1,
                    clipvalue=self.config.model.discriminator.clipvalue,
                    clipnorm=self.config.model.discriminator.clipnorm)
        optimizer = self.process_optimizer(optimizer)

        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model, parallel_model