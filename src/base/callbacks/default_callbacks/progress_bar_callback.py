import sys
from collections import ChainMap
from typing import Literal, TextIO, Callable

from src.base.callbacks import Callback
from src.utils import ProgressBar


class ProgressBarCallback(Callback):
    def __init__(
            self,
            monitors: [str] or str = None,
            monitor_formatters: {str: Callable[[float], str]} = None,
            count_mode: Literal['batch', 'sample'] = 'batch',
            out_stream: TextIO = sys.stdout,
    ):
        super().__init__()

        if isinstance(monitors, str):
            monitors = [monitors]

        self.monitors = monitors or ["average_loss", "accuracy"]

        self.monitor_formatters = monitor_formatters or {}
        self.monitor_formatters.update({
            monitor: lambda val: f'{val:.3f}'
            for monitor in self.monitors
            if monitor not in self.monitor_formatters
        })

        self.count_mode = _check_count_mode(count_mode)
        self.progress_bar: ProgressBar = None

        self.epochs = 1
        self.test_batches = 1
        self.train_batches = 1
        self.batch_size = 100

        self.out_stream = out_stream

    @property
    def test_total(self):
        return self.test_batches if self.count_mode == 'batch' else self.test_batches * self.batch_size

    @property
    def train_total(self):
        return self.train_batches if self.count_mode == 'batch' else self.train_batches * self.batch_size

    @property
    def update_step(self):
        return 1 if self.count_mode == 'batch' else self.batch_size

    def set_state(self, state):
        super().set_state(state)
        self.epochs = state.get('epochs', self.epochs)
        self.test_batches = state.get('test_batches', self.test_batches)
        self.train_batches = state.get('train_batches', self.train_batches)
        self.batch_size = state.get('batch_size', self.batch_size)

    def on_epoch_start(self, epoch, state_dict=None):
        print(f"Epoch: {epoch+1}/{self.epochs}", file=self.out_stream)

    def on_train_epoch_start(self, epoch, state_dict=None):
        if self.train_total > 0:
            self.progress_bar = ProgressBar(
                total=self.train_total,
                prefix="train: ",
                out_stream=self.out_stream,
                fill='#'
            )

    def on_train_batch_end(self, batch_idx, metric_states, state_dict=None):
        merged_metric_dict = dict(ChainMap(*metric_states))
        self.__update_progress_bar(merged_metric_dict)

    def on_train_epoch_end(self, epoch, metric_states, state_dict=None):
        self.__clear_progress_bar()

    def on_test_epoch_start(self, epoch, state_dict=None):
        if self.test_total > 0:
            self.progress_bar = ProgressBar(
                total=self.test_total,
                prefix="test:  ",
                out_stream=self.out_stream,
                fill='#'
            )

    def on_test_batch_end(self, batch_idx, metric_states, state_dict=None):
        merged_metric_dict = dict(ChainMap(*metric_states))
        self.__update_progress_bar(merged_metric_dict)


    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        self.__clear_progress_bar()

    def __clear_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def __update_progress_bar(self, metric_dict):
        if self.progress_bar:
            monitor_items = {monitor: metric_dict[monitor]
                             for monitor in self.monitors
                             if monitor in metric_dict}
            monitors_str = ", ".join(f"{key}: {self.monitor_formatters[key](value)}" for key, value in monitor_items.items())
            self.progress_bar.set_postfix(monitors_str)

            self.progress_bar.update(increase=self.update_step)

def _check_count_mode(count_mode):
    allowed_values = ['batch', 'sample']
    if count_mode not in allowed_values:
        raise ValueError(
            "Invalid value for argument 'count_mode'. "
            f"Allowed {allowed_values}. "
            f"Received: {count_mode}."
        )
    else:
        return count_mode