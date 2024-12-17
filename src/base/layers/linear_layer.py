import numpy as np

from src.base.activation import Activation, linear
from src.base.optimizers.optimizer import Optimizer
from src.base.value_initializer import ValueInitializer, zero_initializer, uniform_initializer
from src.base.layers.layer import Layer


class LinearLayer(Layer):
    def __init__(
            self,
            neurons: int,
            activation : Activation = linear(),
            is_trainable: bool = True,
            prev_weights_initializer: ValueInitializer = uniform_initializer(),

            use_bias: bool = True,
            bias_initializer: ValueInitializer = zero_initializer(),
    ):
        super().__init__(
            neurons=neurons,
            activation=activation,
            prev_weights_initializer=prev_weights_initializer,
            is_trainable=is_trainable
        )
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

        self.bias = None

        self.prev_in = None
        self.prev_s = None
        self.prev_out = None


    # Y = X * W
    def forward(self, in_batch: np.array, training=True) -> np.array:
        self.prev_in = in_batch
        self.prev_s = np.dot(in_batch, self.prev_weights) - (self.bias if self.use_bias else 0)

        self.prev_out = np.apply_along_axis(
                    self.activation,
                    axis=self.prev_s.ndim - 1,
                    arr=self.prev_s,
                )

        return self.prev_out

    def backward(self, layer_gradient: np.array, optimizer: Optimizer):
        de_dy = layer_gradient
        dy_ds = np.apply_along_axis(
                self.activation.jacobian,
                axis=self.prev_s.ndim - 1,
                arr=self.prev_s,
            )

        de_ds = np.einsum('...lj,...lkj->...lk',
                          de_dy,
                          dy_ds,
                          optimize='optimal',
                          )

        self.__adjust_bias(optimizer, de_ds)
        self.__adjust_prev_weights(optimizer, de_ds)

        ds_dx = np.transpose(self.prev_weights)
        de_dx = np.dot(de_ds, ds_dx)

        return de_dx


    def __adjust_prev_weights(self, optimizer: Optimizer, de_ds: np.array):
        if self.prev_weights is None or not self.is_trainable:
            return

        weights_gradient = np.einsum('...lj,...lk->kj',
                                     de_ds,
                                     self.prev_in,
                                     optimize='optimal',
                                     )

        optimizer(param=self.prev_weights, param_gradient=weights_gradient)

    def __adjust_bias(self, optimizer: Optimizer, de_ds: np.array):
        if not self.use_bias or not self.is_trainable:
            return

        bias_gradient = - de_ds.sum(axis=tuple(range(de_ds.ndim - 1)))
        optimizer(param=self.bias, param_gradient=bias_gradient)

    def init_layer_params(self, prev_layer_neurons, reassign_existing=True):
        super().init_layer_params(prev_layer_neurons, reassign_existing=reassign_existing)
        if self.use_bias and (reassign_existing or self.bias is None):
            self.bias = self.bias_initializer(self.neurons)
