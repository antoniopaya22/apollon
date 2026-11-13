from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import intrusion_detection_systems.metrics.show_metrics as s_m


class ShowMetricsTrain(s_m.ShowMetrics):

    # Override
    def operation(self) -> list:
        '''operation

        This method represents the specific behavior that this class has,
        it prints the metrics for the predictions made with the train value.

        Parameters:
            None
        Output:
            list: list
        '''
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        confusion_matrix = metrics.confusion_matrix(
            self._component.y_test, self._component.predictions)
        classification = metrics.classification_report(
            self._component.y_test, self._component.predictions)
        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv)
        self.cv = scores.mean()
        self.std = scores.std()
        '''
        '''

        self.cv = 0
        self.std = 0

        #TODO
        self.precission_m = 0
        self.precission_b = 0

        a = open("./results/show_metrics_y_train.txt", "a")

        a.write(
            f'\n============================== {self._component.dataset} Model Evaluation {self._component} ==============================\n')
        a.write(
            f"Cross Validation Mean Score: scores.{self.cv} with a std {self.std}\n")
        a.write(f"Confusion matrix: {confusion_matrix}\n")
        a.write(f"Classification report: {classification}\n")

        a.close()
        return self.get_list()
    
    def get_list(self) -> list:
        '''get_list

        This method returns the list of the metrics of the model

        Parameters:
            None
        Output:
            list: list
        '''
        return [self.cv, self.std, self.precision_b, self.precision_m]
