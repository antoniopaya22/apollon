from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_validate
import intrusion_detection_systems.metrics.show_metrics as s_m
import numpy as np


class ShowMetricsTest(s_m.ShowMetrics):

    # Override
    def operation(self) -> list:
        """operation

        This method represents the specific behavior that this class has,
        it prints the metrics for the predictions made with the test value.

        Output:
            list: list
        """
        
        n = 5

        cv = ShuffleSplit(n_splits=n, test_size=0.2,
                          random_state=self._component.seed)

        self.confusion_matrix = metrics.confusion_matrix(
            self._component.y_test, self._component.predictions)
        self.classification = metrics.classification_report(
            self._component.y_test, self._component.predictions)

        scoring = ('accuracy', 'roc_auc', 'f1_macro', 'recall_macro')

        scores = cross_validate(self._component.model_trained,
                                self._component.x_train, self._component.y_train, cv=cv, scoring=scoring, return_train_score=True)
        self.cv_accuracy_te = np.mean(scores['test_accuracy'])
        self.std_accuracy_te = np.std(scores['test_accuracy'])

        self.cv_roc_te = np.mean(scores['test_roc_auc'])
        self.std_roc_te = np.std(scores['test_roc_auc'])

        self.cv_f1_te = np.mean(scores['test_f1_macro'])
        self.std_f1_te = np.std(scores['test_f1_macro'])

        self.cv_dr_te = np.mean(scores['test_recall_macro'])
        self.std_dr_te = np.std(scores['test_recall_macro'])

        self.cv_accuracy_tr = np.mean(scores['train_accuracy'])
        self.std_accuracy_tr = np.std(scores['train_accuracy'])

        self.cv_roc_tr = np.mean(scores['train_roc_auc'])
        self.std_roc_tr = np.std(scores['train_roc_auc'])

        self.cv_f1_tr = np.mean(scores['train_f1_macro'])
        self.std_f1_tr = np.std(scores['train_f1_macro'])
        
        self.cv_dr_tr = np.mean(scores['train_recall_macro'])
        self.std_dr_tr = np.std(scores['train_recall_macro'])
        
        self.save_to_file()
        self.print_to_console()

        return self.get_list()

    def save_to_file(self, file="./results/apollon_results.txt"):
        with open(file, "a") as v:
            v.write(
                f'\n============================== {self._component.dataset} Model Evaluation {self._component} ==============================\n')
            v.write(
                f"[TEST]\tCross Validation Mean and std Score for accuracy:\t{self.cv_accuracy_te}\t{self.std_accuracy_te}\n")
            v.write(
                f"[TEST]\tCross Validation Mean and std Score for detection rate:\t{self.cv_dr_te}\t{self.std_dr_te}\n")
            v.write(
                f"[TEST]\tCross Validation Mean and std Score for F1:\t{self.cv_f1_te}\t{self.std_f1_te}\n")
            v.write(
                f"[TEST]\tCross Validation Mean and std Score for roc_auc:\t{self.cv_roc_te}\t{self.std_roc_te}\n\n")
            v.write(
                f"[TRAIN]\tCross Validation Mean and std Score for accuracy:\t{self.cv_accuracy_tr}\t{self.std_accuracy_tr}\n")
            v.write(
                f"[TRAIN]\tCross Validation Mean and std Score for detection rate:\t{self.cv_dr_tr}\t{self.std_dr_tr}\n")
            v.write(
                f"[TRAIN]\tCross Validation Mean and std Score for F1:\t{self.cv_f1_tr}\t{self.std_f1_tr}\n")
            v.write(
                f"[TRAIN]\tCross Validation Mean and std Score for roc_auc:\t{self.cv_roc_tr}\t{self.std_roc_tr}\n\n")
            v.write(f"Confusion matrix: \n{self.confusion_matrix}\n")
            v.write(f"Classification report: \n{self.classification}\n")
            v.write(f"time to train: {self._component.time_total[0]} s\n")
            v.write(f"time to predict: {self._component.time_total[1]} s\n")
            v.write(f"total: {self._component.time_total[2]} s\n")

    def print_to_console(self):
        print(
            f'\n============================== {self._component.dataset} Model Evaluation {self._component} ==============================\n')
        print(
            f"[TEST]\tCross Validation Mean and std Score for F1:\t{self.cv_f1_te}\t{self.std_f1_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for accuracy:\t{self.cv_accuracy_te}\t{self.std_accuracy_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for roc_auc:\t{self.cv_roc_te}\t{self.std_roc_te}")
        print(
            f"[TEST]\tCross Validation Mean and std Score for detection rate:\t{self.cv_dr_te}\t{self.std_dr_te}\n")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for F1:\t{self.cv_f1_tr}\t{self.std_f1_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for accuracy:\t{self.cv_accuracy_tr}\t{self.std_accuracy_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for roc_auc:\t{self.cv_roc_tr}\t{self.std_roc_tr}")
        print(
            f"[TRAIN]\tCross Validation Mean and std Score for detection rate:\t{self.cv_dr_tr}\t{self.std_dr_tr}\n")
        print(f"Confusion matrix: \n{self.confusion_matrix}")
        print(f"Classification report: \n{self.classification}")
        print(f"time to train: {self._component.time_total[0]} s")
        print(f"time to predict: {self._component.time_total[1]} s")
        print(f"total: {self._component.time_total[2]} s")

    def get_list(self) -> list:
        '''get_list

        This method returns the list of the metrics of the model

        Parameters:
            None
        Output:
            list: list
        '''
        return [self._component.time_total[2], self.cv_f1_te, self.std_f1_te, self.cv_accuracy_te, self.std_accuracy_te, self.cv_roc_te, self.std_roc_te, self.cv_dr_te, self.std_dr_te, self.cv_f1_tr, self.std_f1_tr, self.cv_accuracy_tr, self.std_accuracy_tr, self.cv_roc_tr, self.std_roc_tr, self.cv_dr_tr, self.std_dr_tr]
