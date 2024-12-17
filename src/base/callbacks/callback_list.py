from src.base.callbacks import Callback


class CallbackList(Callback):
    def __init__(
            self,
            model=None,
            callbacks: [Callback] = None,
            **kwargs,
    ):
        super().__init__()
        self.__callbacks = callbacks or []
        self.set_model(model)
        self.set_state(kwargs)

    def set_model(self, model):
        self.model = model
        for callback in self.__callbacks:
            callback.set_model(model)

    def set_state(self, state):
        self.state = state
        for callback in self.__callbacks:
            callback.set_state(state)

    def on_fit_start(self, epochs, state_dict=None):
        [callback.on_fit_start(epochs, state_dict)
         for callback in self.__callbacks]

    def on_fit_end(self, state_dict=None):
        [callback.on_fit_end(state_dict)
         for callback in self.__callbacks]

    def on_epoch_start(self, epoch, state_dict=None):
        [callback.on_epoch_start(epoch, state_dict)
         for callback in self.__callbacks]

    def on_epoch_end(self, epoch, state_dict=None):
        [callback.on_epoch_end(epoch, state_dict)
         for callback in self.__callbacks]

    def on_train_epoch_start(self, epoch, state_dict=None):
        [callback.on_train_epoch_start(epoch, state_dict)
         for callback in self.__callbacks]

    def on_train_epoch_end(self, epoch, metric_states, state_dict=None):
        [callback.on_train_epoch_end(epoch, metric_states, state_dict)
         for callback in self.__callbacks]

    def on_test_epoch_start(self, epoch, state_dict=None):
        [callback.on_test_epoch_start(epoch, state_dict)
         for callback in self.__callbacks]

    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        [callback.on_test_epoch_end(epoch, metric_states, state_dict)
         for callback in self.__callbacks]

    def on_train_batch_start(self, batch_idx, state_dict=None):
        [callback.on_train_batch_start(batch_idx, state_dict)
         for callback in self.__callbacks]

    def on_train_batch_end(self, batch_idx, metric_states, state_dict=None):
        [callback.on_train_batch_end(batch_idx, metric_states, state_dict)
         for callback in self.__callbacks]

    def on_test_batch_start(self, batch_idx, state_dict=None):
        [callback.on_test_batch_start(batch_idx, state_dict)
         for callback in self.__callbacks]

    def on_test_batch_end(self, batch_idx, metric_states, state_dict=None):
        [callback.on_test_batch_end(batch_idx, metric_states, state_dict)
         for callback in self.__callbacks]
