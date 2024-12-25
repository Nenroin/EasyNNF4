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
            max_time_steps=5,
            predict_modes: [Literal['all', 'test', 'train']] = None,
            loss_title_mode: Literal['all', 'test', 'train'] = 'test'
    ):
        input_neurons = data_loader.input_neurons

        abscissas = data_loader.abscissas
        ordinates = data_loader.ordinates

        predict_modes = predict_modes or ['all']
        predicts_params = []

        if 'all' in predict_modes or 'all' in loss_title_mode:
            predicts_params.append({
                'mode': 'all',
                'test_abscissas': data_loader.get_abscissas(),
                'test_ordinates': data_loader.get_ordinates(),
                'predict_color': 'r',
            })

        if 'test' in predict_modes or 'test' in loss_title_mode:
            predicts_params.append({
                'mode': 'test',
                'test_abscissas': data_loader.get_test_abscissas(),
                'test_ordinates': data_loader.get_test_ordinates(),
                'predict_color': 'm',
            })

        if 'train' in predict_modes or 'train' in loss_title_mode:
            predicts_params.append({
                'mode': 'train',
                'test_abscissas': data_loader.get_train_abscissas(),
                'test_ordinates': data_loader.get_train_ordinates(),
                'predict_color': 'k',
            })

        plt.figure(figsize=(12, len(models) * 4))
        for i, model in enumerate(models):
            plt.subplot(len(models), 1, i + 1)
            plt.axvline(x=data_loader.get_test_abscissas()[0], color='g', label='test')

            average_loss = 0

            for predict_params in predicts_params:
                mode, test_abscissas, test_ordinates, predict_color = predict_params.values()

                nn_output, av_loss = ResultPlotter.get_predicted_data(model, test_ordinates)

                if mode == loss_title_mode:
                    average_loss = av_loss

                predict_params['test_ordinates'] = nn_output

            plt.title(model.name + f'   average_{loss_title_mode}_loss = {average_loss}')
            plt.plot(abscissas, ordinates, 'b')

            for predict_params in predicts_params:
                mode, test_abscissas, test_ordinates, predict_color = predict_params.values()
                plt.plot(test_abscissas, test_ordinates, predict_color, linestyle='dashed')

        plt.show()

    @staticmethod
    def get_predicted_data(model, ordinates, max_timesteps=999999):
        input_neurons = model.layers[0].neurons

        def build_sequences(nn_output_values):
            nn_sequences = []
            for j in range(input_neurons, len(nn_output_values) + 1):
                nn_sequences.append(nn_output_values[j - input_neurons:j])
            return np.expand_dims(np.array(nn_sequences), axis=0)


        overall_loss = 0
        iterations = 0

        nn_output = list(ordinates[:input_neurons])
        for j in range(input_neurons, len(ordinates)):
            nn_in_vector = build_sequences(nn_output)[:, -max_timesteps:, :]
            nn_output_vector = model.forward(nn_in_vector)
            nn_output.append(nn_output_vector.item((0, -1, 0)))
            overall_loss += np.power(nn_output[-1] - ordinates[j], 2).sum() / 2
            iterations += 1

        average_loss = overall_loss / iterations

        return nn_output, average_loss