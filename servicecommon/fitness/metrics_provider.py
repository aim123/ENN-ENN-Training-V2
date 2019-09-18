

class MetricsProvider:
    """
    An interface for dealing with an entity (like an Individual) which has
    a way of giving Metrics Records.
    """

    def get_metrics(self):
        """
        Returns the metrics of this entity.
        :return: a dictionary of metrics
        """
        raise NotImplementedError
