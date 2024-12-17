import numpy as np

from src.base.activation import linear
from src.base.optimizers.optimizer import Optimizer
from src.base.value_initializer import none_weights_initializer
from src.base.layers.layer import Layer

class InputLayer(Layer):
    def __init__(
            self,
            neurons: int,
    ):
        super().__init__(
            neurons=neurons,
            activation=linear(),
            prev_weights_initializer=none_weights_initializer(),
            is_trainable=False,
        )

    # Y = X * W
    def forward(self, in_batch: np.array, training=True) -> np.array:
        return in_batch

    def backward(self, layer_gradient: np.array, optimizer: Optimizer):
        return layer_gradient
