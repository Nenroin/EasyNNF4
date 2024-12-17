import numpy as np

class DataBatchWrapper:
    def __init__(self, data, batch_size):
        data = (np.array(data[0]), np.array(data[1]))
        self.data_x, self.data_y = (data[0] if data[0].ndim > 1 else data[0][..., None],
                                    data[1] if data[1].ndim > 1 else data[1][..., None])
        self.batch_size = batch_size
        self.index = 0

    def __len__(self):
        return self.data_x.shape[0] // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index + self.batch_size > len(self.data_x):
            raise StopIteration

        batch = (self.data_x[self.index: self.index + self.batch_size],
                 self.data_y[self.index: self.index + self.batch_size])
        self.index += self.batch_size
        return batch

    def __getitem__(self, idx):
        idx *= self.batch_size
        batch = (self.data_x[idx: self.index + self.batch_size],
                 self.data_y[idx: self.index + self.batch_size])
        return batch
