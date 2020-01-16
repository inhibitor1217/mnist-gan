from keras import Input, Model
from keras.layers import Conv2DTranspose, BatchNormalization

from base.base_model import BaseModel

class Generator(BaseModel):
    def define_model(self, model_name):
        _input = Input(shape=(1, 1, 64), name=f'{model_name}_input')

        x = Conv2DTranspose(filters=16, kernel_size=3, strides=4, 
                        activation='relu', name=f'{model_name}_deconv_1')(_input)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=8, kernel_size=3, strides=2,
                        activation='relu', name=f'{model_name}_deconv_2')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=4, kernel_size=3, strides=2,
                        activation='relu', name=f'{model_name}_deconv_3')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=2,
                        activation='tanh', name=f'{model_name}_deconv_4')(x)

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, model_name):
        raise NotImplementedError