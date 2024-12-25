from collections import ChainMap
from typing import Literal

from src.base.callbacks import Callback

class EarlyStoppingCallback(Callback):
    def __init__(
            self,
            mode: Literal['min', 'max'] = 'min',
            monitor: str = "average_loss",
            min_abs_delta: float = 0,
            patience: int = 0,
            start_from_epoch: int = 0,
            compare_with: Literal['best', 'previous'] = 'best',
            show_message: bool = True,
    ):
        super().__init__()

        self.monitor = monitor
        self.mode = mode
        self.min_abs_delta = min_abs_delta
        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.compare_with = compare_with
        self.show_message = show_message

        self.__process_mode(mode)
        self.__process_compare_with(compare_with)

        self.prev_monitor_value = None
        self.best_monitor_value = None
        self.epochs_without_improvement = 0

    def __process_mode(self, mode):
        if mode not in ['max', 'min']:
            raise ValueError(f'mode must be either "max" or "min", given: {mode}')
        elif mode == 'max':
            def comparator(monitor_val, reference_val):
                delta = monitor_val - reference_val
                return delta > 0 and abs(delta) >= self.min_abs_delta

            self.has_monitor_improved = comparator
        elif mode == 'min':
            def comparator(monitor_val, reference_val):
                delta = monitor_val - reference_val
                return delta < 0 and abs(delta) >= self.min_abs_delta

            self.has_monitor_improved = comparator

    def __process_compare_with(self, compare_with):
        if compare_with not in ['best', 'previous']:
            raise ValueError(f'compare_with must be either "best" or "previous", given: {compare_with}')
        elif compare_with == 'previous':
            self.referenced_val = lambda : self.prev_monitor_value
        elif compare_with == 'best':
            self.referenced_val = lambda : self.best_monitor_value

    def on_fit_start(self, epochs, state_dict=None):
        self.prev_monitor_value = None
        self.best_monitor_value = None
        self.epochs_without_improvement = 0

    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        monitor_val = self.__get_monitor_value(metric_states)

        if monitor_val is None:
            return

        if epoch == 0 or epoch < self.start_from_epoch:
            self.prev_monitor_value = monitor_val
            self.best_monitor_value = monitor_val
            return

        self.epochs_without_improvement += 1
        if self.has_monitor_improved(monitor_val, self.referenced_val()):
            self.epochs_without_improvement = 0

        if monitor_val < 0.0009:
            self.model.stop_training = True
            if self.show_message:
                print(f'Early stopping at epoch {epoch+1}')

        self.prev_monitor_value = monitor_val

        if self.has_monitor_improved(monitor_val, self.best_monitor_value):
            self.best_monitor_value = monitor_val

    def __get_monitor_value(self, metric_states):
        merged_state = dict(ChainMap(*metric_states))
        if self.monitor in merged_state:
            return merged_state[self.monitor]
        return None
