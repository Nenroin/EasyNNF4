import pickle
from abc import ABC, abstractmethod

import numpy as np

from src.base.callbacks import Callback
from src.base.callbacks.callback_list import CallbackList
from src.base.data.data_batch_wrapper import DataBatchWrapper
from src.base.data.model_data_source import ModelDataSource
from src.base.layers.layer import Layer
from src.base.loss_function import LossFunction
from src.base.metrics import MetricList
from src.base.metrics.metric import Metric
from src.base.optimizers.optimizer import Optimizer


class Model(ABC):
    def __init__(self,
                 name: str,
                 layers: [Layer] = None,
                 loss_function: LossFunction = None,
                 optimizer: Optimizer = None,
                 metrics: [Metric] or MetricList = None,
                 ):
        self.name = name

        self.optimizer = optimizer
        self.loss_function = loss_function

        if not isinstance(metrics, MetricList):
            self.metrics = MetricList(metrics)

        self.layers = layers or []

        self.stop_training = False

    def build(
            self,
            loss_function: LossFunction = None,
            optimizer: Optimizer = None,
            metrics: [Metric] or MetricList = None,
            reassign_existing_weights: bool = True,
    ):
        self.loss_function = loss_function or self.loss_function
        self.optimizer = optimizer or self.optimizer
        if metrics is not None and not isinstance(metrics, MetricList):
            self.metrics = MetricList(metrics)

        self.init_layers_params(reassign_existing=reassign_existing_weights)

    @abstractmethod
    def fit(self, model_data_source: ModelDataSource, epochs = 1, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def train_epoch(self, train_data: DataBatchWrapper, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def test_epoch(self, test_data: DataBatchWrapper, callbacks: CallbackList or [Callback] = None):
        pass

    @abstractmethod
    def forward(self, in_batch: np.array, training=True):
        pass

    @abstractmethod
    def backward(self, loss_gradient: np.array):
        pass

    @abstractmethod
    def add_layer(self, layer: Layer):
        pass

    @abstractmethod
    def init_layers_params(self, reassign_existing=True):
        pass

    @abstractmethod
    def summary(self):
        pass
