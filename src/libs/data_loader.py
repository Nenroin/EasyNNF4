import math
import numpy as np
from src.base.data import ModelDataSource


def function(in_data):
    return np.sin(in_data)


class DataLoader:
    def __init__(
            self,
            input_neurons=3,
            train_dataset_size=512,
            test_dataset_size=128,
            periods_in_train_dataset=10,
            time_steps=4,
            batch_size=1,
    ):
        period = 2 * math.pi

        def reduce_to_max_divisible(num, divisor):
            return (num // divisor) * divisor

        train_dataset_size = reduce_to_max_divisible(train_dataset_size, batch_size * time_steps)
        test_dataset_size = reduce_to_max_divisible(test_dataset_size, batch_size * time_steps)

        self.input_neurons = input_neurons
        self.periods_in_train_dataset = periods_in_train_dataset

        self.time_steps = time_steps
        self.batch_size = batch_size

        self.train_dataset_size = train_dataset_size
        self.train_points_count = train_dataset_size + input_neurons

        self.test_dataset_size = test_dataset_size
        self.test_points_count = test_dataset_size + input_neurons

        self.dataset_size = self.train_dataset_size + self.test_dataset_size
        self.dataset_points_count = self.train_points_count + self.test_points_count

        self.abscissas = np.array([periods_in_train_dataset * x * period / (self.train_points_count - 1)
                                   for x in range(self.dataset_points_count)])

        self.ordinates = function(self.abscissas)

        x_train, y_train = self.__calculate_train_values()
        x_test, y_test = self.__calculate_test_values()

        self.data_source = ModelDataSource(
            train_data=(x_train, y_train),
            test_data=(x_test, y_test),
            batch_size=self.batch_size,
        )

    def __calculate_train_values(self):
        y_train_function = self.ordinates[:self.train_points_count]

        x_train, y_train = self.__extract_sequences(y_train_function)

        return x_train, y_train

    def __calculate_test_values(self):
        y_test_function = self.ordinates[self.train_points_count:]

        x_test, y_test = self.__extract_sequences(y_test_function)

        return x_test, y_test

    def __extract_sequences(self, data):
        nn_in = []
        nn_out = []

        for i in range(self.input_neurons, len(data)):
            nn_in.append(data[i - self.input_neurons:i])
            nn_out.append(data[i])

        nn_in = np.array(nn_in).reshape((-1, self.time_steps, self.input_neurons))
        nn_out = np.array(nn_out).reshape((-1, self.time_steps, 1))

        return nn_in, nn_out

    def show_stage_1(self):
        print("Stage 1: data preparing")
        for i in range(self.train_dataset_size + self.test_dataset_size):
            print(f"x_{i + 1}={self.abscissas[i]}, ", end=" ")
            print(f"y_{i + 1}={self.ordinates[i]}")

    def show_stage_2(self):
        print("Stage 2: split data on train/test data")
        for i in range(self.train_dataset_size + self.test_dataset_size):
            if i == 0:
                print("train_data:")
            elif i == self.train_dataset_size:
                print("test_data:")
            print(f"x_{i + 1}={self.abscissas[i]}, ", end=" ")
            print(f"y_{i + 1}={self.ordinates[i]}")

    def show_stage_3(self):
        print("Stage 3: prepare train/test data for NN")
        b = False

        for y_idx in range(len(self.ordinates) - self.input_neurons):
            if y_idx == self.train_dataset_size - self.input_neurons:
                print("test_data (y60 -> original value, y’61 -> model output value):")
                b = True

            values = [f"y_{y_idx + 1 + i}={self.ordinates[y_idx + i]:.4f}" for i in range(self.input_neurons)]
            prefix = " -> y'" if b else " -> y_"
            values.append(f"{prefix}{y_idx + 1 + self.input_neurons}={self.ordinates[y_idx + self.input_neurons]:.4f}")
            print(" ".join(values))

            if b: return

    def get_data_source(self):
        return self.data_source

    def get_abscissas(self):
        return self.abscissas

    def get_ordinates(self):
        return self.ordinates

    def get_test_abscissas(self):
        return self.abscissas[self.train_points_count:]

    def get_test_ordinates(self):
        return self.ordinates[self.train_points_count:]

    def get_train_abscissas(self):
        return self.abscissas[:self.train_points_count]

    def get_train_ordinates(self):
        return self.ordinates[:self.train_points_count]
