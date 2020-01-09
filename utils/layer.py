from typing import Optional

from keras.engine import Layer
from keras.layers import Lambda


def named(name: str) -> Layer:
    return Lambda(lambda x: x, name=name)