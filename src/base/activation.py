from typing import Callable
import numpy as np

class Activation:
    def __init__(self,
                 activation_fn: Callable[[np.array], np.array],
                 jacobian_fn: Callable[[np.array], np.array]
                 ):
        self.__activation_fn = activation_fn
        self.__jacobian_fn = jacobian_fn

    def __call__(self, x_vec: np.array) -> np.array:
        return self.__activation_fn(x_vec)

    def jacobian(self, x_vec: np.array) -> np.array:
        return self.__jacobian_fn(x_vec)

def relu() -> Activation:
    def call(x: np.array) -> np.array:
        return np.maximum(x, 0)

    def jacobian(x: np.array) -> np.array:
        x = np.array(x)
        jacobian__ = np.diag(np.where(x <= 0, 0, 1))
        return jacobian__

    return Activation(call, jacobian)

def linear() -> Activation:
    def call(x: np.array) -> np.array:
        return x

    def jacobian(x: np.array) -> np.array:
        x = np.array(x)
        jacobian_ = np.diag(np.ones(len(x)))
        return jacobian_

    return Activation(call, jacobian)

def softmax() -> Activation:
    def call(s: np.array) -> np.array:
        z = s - s.max()
        return np.exp(z) / sum(np.exp(z))

    def jacobian(s: np.array) -> np.array:
        jacobian_matrix = np.diagflat(s) + np.einsum('i,j->ij',
                                          s, s,
                                          optimize='optimal')
        return jacobian_matrix

    return Activation(call, jacobian)

def sigmoid() -> Activation:
    def call(x: np.array) -> np.array:
        return np.array(1 / (1 + np.exp(-x)))

    def jacobian(x: np.array) -> np.array:
        sigmoid_arr = call(x)
        jacobian_ = np.diag(np.multiply(sigmoid_arr, (1 - sigmoid_arr)))
        return jacobian_

    return Activation(call, jacobian)

def arctan() -> Activation:
    def call(x: np.array) -> np.array:
        return np.arctan(x)

    def jacobian(x: np.array) -> np.array:
        jacobian_ = np.diag(np.array([1 / (1 + xi**2) for xi in x]))
        return jacobian_

    return Activation(call, jacobian)
