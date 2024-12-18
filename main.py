from src.base.activation import linear, sigmoid
from src.base.callbacks.default_callbacks import ProgressBarCallback
from src.base.callbacks.default_callbacks.early_stopping_callback import EarlyStoppingCallback
from src.base.layers import LinearLayer, InputLayer
from src.base.layers.elman_layer import ElmanLayer
from src.base.loss_function import mse
from src.base.metrics import LossMetric
from src.base.models import SequentialModel
from src.base.optimizers import Adam
from src.libs.data_loader import DataLoader
from src.libs.result_plotter import ResultPlotter

train_dataset_size = 128
test_dataset_size = 64
input_neurons = 8
batch_size = 1
periods_in_train_dataset = 1

lr = 0.001
epochs=500
stacked_layers = 1
state_size = 4
time_steps = 4


data_loader = DataLoader(
    train_dataset_size=train_dataset_size,
    test_dataset_size=test_dataset_size,

    input_neurons=input_neurons,
    time_steps=time_steps,
    batch_size=batch_size,
    periods_in_train_dataset=periods_in_train_dataset,
)

data_loader.show_stage_1()
print()
data_loader.show_stage_2()
print()
data_loader.show_stage_3()
print()
print("Stage 4: train & test model")

data_source = data_loader.get_data_source()

model = SequentialModel(
    layers=[
        InputLayer(input_neurons),
        ElmanLayer(state_size, activation=sigmoid(), stacked_layers=stacked_layers),
        LinearLayer(1, activation=linear())
    ]
)

model.build(
    loss_function=mse(),
    optimizer=Adam(learning_rate=lr),
    metrics=[LossMetric(loss_function=mse())],
)

model.fit(
    model_data_source=data_source,
    epochs=epochs,
    callbacks=[
        ProgressBarCallback(
            monitors=['average_loss'],
            monitor_formatters={'average_loss': lambda val: f'{val}'}
        ),
        EarlyStoppingCallback(
            mode='min',
            monitor='average_loss',
            compare_with='previous',
            patience=10,
            start_from_epoch=10,
        )
    ],
)

ResultPlotter.visualize_models(
    data_loader=data_loader,
    models=[
        model
    ],
    max_time_steps=time_steps,
    predict_modes=[
        'test',
    ],
    loss_title_mode='test'
)
