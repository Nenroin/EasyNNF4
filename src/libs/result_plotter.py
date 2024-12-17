from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
from src.libs.data_loader import DataLoader
from src.base.models import Model

class ResultPlotter:
    @staticmethod
    def visualize_models(
            data_loader: DataLoader,
            models: [Model],
            max_timesteps = 5,
            predict_mode: Literal['all', 'test', 'train'] = 'all',
    ):
        input_neurons = data_loader.input_neurons

        abscissas = data_loader.abscissas
        ordinates = data_loader.ordinates

        if predict_mode == 'all':
            test_abscissas = data_loader.get_abscissas()
            test_ordinates = data_loader.get_ordinates()
        elif predict_mode == 'test':
            test_abscissas = data_loader.get_test_abscissas()
            test_ordinates = data_loader.get_test_ordinates()
        elif predict_mode == 'train':
            test_abscissas = data_loader.get_train_abscissas()
            test_ordinates = data_loader.get_train_ordinates()


        plt.figure(figsize=(12, len(models) * 4))
        for i, model in enumerate(models):
            plt.subplot(len(models), 1, i + 1)
            plt.axvline(x=data_loader.get_test_abscissas()[0], color='g', label='test')

            def build_sequences(nn_output_values):
                nn_sequences = []
                for j in range(input_neurons, len(nn_output_values) + 1):
                    nn_sequences.append(nn_output_values[j - input_neurons:j])
                return np.expand_dims(np.array(nn_sequences), axis=0)

            overall_loss = 0
            iterations = 0

            nn_output = list(test_ordinates[:input_neurons])
            for j in range(input_neurons, len(test_abscissas)):
                nn_in_vector = build_sequences(nn_output)[:, -max_timesteps:, :]
                nn_output_vector = model.forward(nn_in_vector)
                nn_output.append(nn_output_vector.item((0, -1, 0)))

                overall_loss += np.power(nn_output[-1] - test_ordinates[j], 2).sum() / 2
                iterations += 1

            plt.title(model.name + f'   average_loss = {overall_loss/iterations:.3e}')
            plt.plot(abscissas, ordinates, 'b')
            plt.plot(test_abscissas, nn_output, 'r', linestyle='dashed')

        plt.show()
