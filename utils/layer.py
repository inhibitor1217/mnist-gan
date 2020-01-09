from typing import Optional

from tensorflow.keras.engine import Layer
from tensorflow.keras.layers import Lambda


def named(name: str) -> Layer:
    return Lambda(lambda x: x, name=name)