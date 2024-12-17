from src.base.callbacks import Callback


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, epochs, state_dict=None):
        print(f"on_fit_start, epochs:{epochs}")

    def on_fit_end(self, state_dict=None):
        print(f"on_fit_end")

    def on_epoch_start(self, epoch, state_dict=None):
        print(f"on_epoch_start, epoch:{epoch}")

    def on_epoch_end(self, epoch, state_dict=None):
        print(f"on_epoch_end, epoch:{epoch}")

    def on_train_epoch_start(self, epoch, state_dict=None):
        print(f"on_train_epoch_start, epoch:{epoch}")

    def on_train_epoch_end(self, epoch, metric_states, state_dict=None):
        print(f"on_train_epoch_end, epoch:{epoch}, metric_states:{metric_states}")

    def on_test_epoch_start(self, epoch, state_dict=None):
        print(f"on_test_epoch_start, epoch:{epoch}")

    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        print(f"on_test_epoch_end, epoch:{epoch}, metric_states:{metric_states}")

    def on_train_batch_start(self, batch_idx, state_dict=None):
        print(f"on_train_batch_start, batch_idx:{batch_idx}")

    def on_train_batch_end(self, batch_idx, metric_states, state_dict=None):
        print(f"on_train_batch_end, batch_idx:{batch_idx}, metric_states:{metric_states}")

    def on_test_batch_start(self, batch_idx, state_dict=None):
        print(f"on_test_batch_start, batch_idx:{batch_idx}")

    def on_test_batch_end(self, batch_idx, metric_states, state_dict=None):
        print(f"on_test_batch_end, batch_idx:{batch_idx}, metric_states:{metric_states}")
