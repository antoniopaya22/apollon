import shared.models.model as m


class ShowMetrics(m.Model):
    def __init__(self, component: m.Model) -> None:
        """__init__

        This method is used to initialize the ShowMetrics class.

        Parameters:
            component: Model class
        Output:
            None
        """
        super().__init__(component.x_train, component.y_train,
                         component.x_test, component.y_test, component.dataset, component.seed)
        self._component = component
        self.accuracy_test = 0
        self.accuracy_train = 0
        self.time = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.cv = 0
        self.std = 0
        self.precision_b = 0
        self.precision_m = 0

    def get_list(self) -> list:
        """get_list

        This method returns the list of the metrics of the model

        Output:
            list: list
        """
        pass

    @property
    def model(self) -> m.Model:
        """show_metrics.py

        This method defines the model to be used and is added to the class variables

        Output:
            Model: Model
        """
        return self._component
