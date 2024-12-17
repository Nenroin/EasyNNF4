from src.utils import not_overridden


class Callback:
    def __init__(
            self,
    ):
        self.model = None
        self.state = None

    def set_state(self, state):
        self.state = state

    def set_model(self, model):
        self.model = model

    @not_overridden
    def on_fit_start(self, epochs, state_dict=None):
        pass

    @not_overridden
    def on_fit_end(self, state_dict=None):
        pass

    @not_overridden
    def on_epoch_start(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_epoch_end(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_train_epoch_start(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_train_epoch_end(self, epoch, metric_states: [{}], state_dict=None):
        pass

    @not_overridden
    def on_test_epoch_start(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_test_epoch_end(self, epoch, metric_states: [{}], state_dict=None):
        pass

    @not_overridden
    def on_train_batch_start(self, batch, state_dict=None):
        pass

    @not_overridden
    def on_train_batch_end(self, batch, metric_states: [{}], state_dict=None):
        pass

    @not_overridden
    def on_test_batch_start(self, batch, state_dict=None):
        pass

    @not_overridden
    def on_test_batch_end(self, batch, metric_states: [{}], state_dict=None):
        pass
