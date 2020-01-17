from keras import Input, Model
from keras.layers import Conv2DTranspose, BatchNormalization, ReLU

from base.base_model import BaseModel

class Generator(BaseModel):
    def define_model(self, model_name):
        _input = Input(shape=(1, 1, 64), name=f'{model_name}_input')

        x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, output_padding=0,
                        use_bias=False, name=f'{model_name}_deconv_1',)(_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters=4, kernel_size=3, strides=3, output_padding=0,
                        use_bias=False, name=f'{model_name}_deconv_2')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=3, output_padding=1,
                        activation='tanh', name=f'{model_name}_deconv_3')(x)

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, model_name):
        raise NotImplementedError