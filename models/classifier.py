from base.base_model import BaseModel

from keras import Model, Input
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Flatten, Dense, Softmax
from keras.optimizers import Adam

class Classifier(BaseModel):
    def define_model(self, model_name):
        def conv_block(_input, filters, kernel_size, strides, name_prefix):
            _x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                        name=f'{name_prefix}conv')(_input)
            _x = BatchNormalization()(_x)
            _x = ReLU()(_x)
            _x = MaxPool2D(pool_size=2)(_x)
            return _x
        
        _input = Input(shape=(28, 28, 1), name=f'{model_name}_input') # (28, 28, 1)

        x = conv_block(_input, filters=4, kernel_size=3, strides=1, name_prefix='conv_block_1') # (14, 14, 4)
        x = conv_block(x,      filters=8, kernel_size=3, strides=1, name_prefix='conv_block_2') # (7,  7,  8)
        x = Flatten()(x) # (392,)
        x = Dense(64)(x) # (64,)
        x = Dense(10)(x) # (10,)
        x = Softmax()(x)

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, model_name):
        model = self.define_model(model_name)

        optimizer = Adam(lr=self.config.model.lr, beta_1=self.config.model.beta1,
                    beta_2=self.config.model.beta2, clipvalue=self.config.model.clipvalue,
                    clipnorm=self.config.model.clipnorm)
        optimizer = self.process_optimizer(optimizer)

        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model, parallel_model