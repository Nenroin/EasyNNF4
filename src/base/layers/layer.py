from src.base.activation import *
from abc import ABC, abstractmethod
from src.base.optimizers.optimizer import Optimizer
from src.base.value_initializer import ValueInitializer


class Layer(ABC):
    def __init__(
            self,
            neurons: int,
            activation: Activation = None,
            is_trainable: bool = True,
            prev_weights_initializer: ValueInitializer = None
    ):
        self.neurons = neurons
        self.activation = activation
        self.prev_weights_initializer = prev_weights_initializer
        self.is_trainable = is_trainable

        self.prev_weights = None

    @abstractmethod
    def forward(self, in_batch: np.array, training=True) -> np.array:
        pass

    @abstractmethod
    def backward(self, layer_gradient: np.array, optimizer: Optimizer) -> np.array:
        pass

    def init_layer_params(self, prev_layer_neurons, reassign_existing=True):
        if reassign_existing or self.prev_weights is None:
            self.prev_weights = self.prev_weights_initializer((prev_layer_neurons, self.neurons))
