from src.base.metrics import Metric


class MetricList(Metric):
    def __init__(
            self,
            metrics: [Metric] = None,
    ):
        super(MetricList, self).__init__()

        self.metrics = metrics

    def update_state(self, y_pred, e):
        if self.metrics:
            for metric in self.metrics:
                metric.update_state(y_pred, e)

    def clear_state(self):
        if self.metrics:
            for metric in self.metrics:
                metric.clear_state()

    def get_metric_state(self):
        if self.metrics:
            states = []
            for metric in self.metrics:
                states.append(metric.get_metric_state())
            return states

    def get_published_value(self):
        if self.metrics:
            values = []
            for metric in self.metrics:
                values.append(metric.get_published_value())
            return values

    def __iter__(self):
        if self.metrics:
            return self.metrics.__iter__()