import pickle

from collections import ChainMap
from typing import Literal

from src.base.callbacks import Callback

class ModelSaveCallback(Callback):
    def __init__(
            self,
            monitor: str,
            monitor_save_threshold: float,
            filepath: str,
            mode: Literal['min', 'max'] = 'min',
    ):
        super().__init__()

        self.monitor = monitor
        self.monitor_save_threshold = monitor_save_threshold
        self.filepath = filepath

        if mode not in ['max', 'min']:
            raise ValueError(f'mode must be either "max" or "min", given: {mode}')
        elif mode == 'max':
            def comparator(monitor_val):
                if monitor_val > self.monitor_save_threshold:
                    return True if self.best_monitor_value is None else monitor_val > self.best_monitor_value
                return False

            self.comparator = comparator
        elif mode == 'min':
            def comparator(monitor_val):
                if monitor_val < self.monitor_save_threshold:
                    return True if self.best_monitor_value is None else monitor_val < self.best_monitor_value
                return False

            self.comparator = comparator


        self.best_monitor_value = None
        self.best_model = None

    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        merged_state = dict(ChainMap(*metric_states))
        if self.monitor in merged_state and isinstance(merged_state[self.monitor], float):
            monitor = merged_state[self.monitor]
            if self.comparator(monitor):
                self.best_monitor_value = monitor
                self.best_model = pickle.dumps(self.model)

    def on_fit_end(self, state_dict=None):
        if self.best_model is not None:
            with open(self.filepath, 'wb') as file:
                file.write(self.best_model)
