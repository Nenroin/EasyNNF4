import numpy as np
from src.base.data.data_batch_wrapper import DataBatchWrapper


class ModelDataSource:
    def __init__(
            self,
            train_data: (np.array, np.array) = None,
            test_data: (np.array, np.array) = None,
            batch_size=1,
    ):
        self.__test_data = test_data or ([], [])
        self.__train_data = train_data or ([], [])

        self.batch_size = batch_size

    def train_data_batches(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return DataBatchWrapper(self.__train_data, batch_size)

    def test_data_batches(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return DataBatchWrapper(self.__test_data, batch_size)
