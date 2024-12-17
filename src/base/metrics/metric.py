from abc import abstractmethod, ABC


class Metric(ABC):
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def update_state(self, y_pred, e):
        pass

    @abstractmethod
    def clear_state(self):
        pass

    @abstractmethod
    def get_metric_state(self):
        pass

    @abstractmethod
    def get_published_value(self):
        pass