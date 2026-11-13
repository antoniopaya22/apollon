from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import intrusion_detection_systems.metrics.show_metrics as s_m


class ShowMetricsLoadModel(s_m.ShowMetrics):

    # Override
    def operation(self, test_x: list = None, test_y: list = None) -> list:
        '''operation

        This method represents the specific behavior that this class has,
        it prints the metrics for the predictions made with the test value.

        Parameters:
            test: list
            train: list
        Output:
            list: list
        '''
        print("ShowMetricsLoadModel")
        
        pred = self._component.predict(test_x)
        
        test_x = test_x if test_x is not None else self._component.y_train
        test_y = test_y if test_y is not None else self._component.y_test
        
        self.accuracy_test = accuracy_score(test_y, pred)

        self.recall = recall_score(test_y, pred, average='weighted')

        self.precision = precision_score(test_y, pred, average='weighted')

        self.f1s = f1_score(test_y, pred, average='weighted')

        self.time = self._component.time_total[2]

        confusion_matrix = metrics.confusion_matrix(
            test_y, pred)
        classification = metrics.classification_report(
            test_y, pred)

        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        scoring = 'neg_log_loss'

        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv, scoring=scoring)
        self.cv_log_loss = scores.mean()
        self.std_log_loss = scores.std()

        scoring = 'accuracy'

        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv, scoring=scoring)
        self.cv_accuracy = scores.mean()
        self.std_accuracy = scores.std()

        scoring = 'roc_auc'

        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv, scoring=scoring)
        self.cv_roc_auc = scores.mean()
        self.std_roc_auc = scores.std()

        scoring = 'neg_mean_squared_error'

        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv, scoring=scoring)
        self.cv_neg_mean_squared_error = scores.mean()
        self.std_neg_mean_squared_error = scores.std()

        scoring = 'r2'

        scores = cross_val_score(self._component.model_trained,
                                 self._component.x_train, self._component.y_train, cv=cv, scoring=scoring)
        self.cv_r2 = scores.mean()
        self.std_r2 = scores.std()

        v = open("./results/show_metrics_load_model.txt", "a")

        v.write(
            f'\n============================== {self._component.dataset} Model Evaluation {self._component} ==============================\n')
        v.write(f"Accuracy: {self.accuracy_test}\n")
        v.write(f"Recall: {self.recall}\n")
        v.write(f"Precision: {self.precision}\n")
        v.write(f"F1-Score: {self.f1s}\n")
        v.write(
            f"Cross Validation Mean Score for neg_log_loss: scores.{self.cv_log_loss} with a std {self.std_log_loss}\n")
        v.write(
            f"Cross Validation Mean Score for accuracy: scores.{self.cv_accuracy} with a std {self.std_accuracy}\n")
        v.write(
            f"Cross Validation Mean Score for roc_auc: scores.{self.cv_roc_auc} with a std {self.std_roc_auc}\n")
        v.write(
            f"Cross Validation Mean Score for neg_mean_squared_error: scores.{self.cv_neg_mean_squared_error} with a std {self.std_neg_mean_squared_error}\n")
        v.write(
            f"Cross Validation Mean Score for r2: scores.{self.cv_r2} with a std {self.std_r2}\n")
        v.write(f"Confusion matrix: {confusion_matrix}\n")
        v.write(f"Classification report: {classification}\n")
        v.write(f"time to train: {self._component.time_total[0]} s\n")
        v.write(f"time to predict: {self._component.time_total[1]} s\n")
        v.write(f"total: {self.time} s\n")

        v.close()

        g = open("./results/loss.txt", "a")
        g.write(str(self.cv_log_loss) + "\n")
        g.close()

        return self.get_list()

    def get_list(self) -> list:
        '''get_list

        This method returns the list of the metrics of the model

        Parameters:
            None
        Output:
            list: list
        '''
        return [self.accuracy_test, self.time, self.recall, self.precision, self.f1s, self.cv_log_loss, self.std_log_loss, self.cv_accuracy, self.std_accuracy, self.cv_roc_auc, self.std_roc_auc, self.cv_neg_mean_squared_error, self.std_neg_mean_squared_error, self.cv_r2, self.std_r2]
