from typing import Callable
import numpy as np


class MatchingFunction:
    def __init__(self,
                 matching_fn: Callable[[np.array, np.array], bool] = None,
                 ):
        self.__matching_function = matching_fn

    def __call__(self, y_pred: np.array, e: np.array) -> np.array:
        return self.__matching_function(y_pred, e)


def one_hot_matching_function():
    def matching_function(y_pred, e):
        return y_pred.argmax() == e.argmax()

    return MatchingFunction(matching_fn=matching_function)
