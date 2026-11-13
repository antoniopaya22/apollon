import pandas as pd
import numpy as np
import intrusion_detection_systems.metrics.show_metrics as s_m
from intrusion_detection_systems.metrics import SMLM


class ShowAttackBlackBox(s_m.ShowMetrics):

    # Override
    def operation(self) -> list:
        '''operation

        This method represents the specific behavior that this class has,
        it prints the confusion matrix for the predictions made with the test value.

        Parameters:
            None
        Output:
            list: list
        '''
        print("-" * 100)
        self.train_losses = []
        self.test_losses = []

        smlm = SMLM(self._component.model_trained)

        x, y = self.load_data()

        self.train_losses = smlm.operation()[5]
        self.test_losses = smlm.operation(test_x=x, test_y=y)[5]

        v = open("./results/show_metrics_attack_black_box.txt", "a")
        v.write(f"train_losses: {self.train_losses}\n")
        v.write(f"test_losses: {self.test_losses}\n")
        v.close()

        return self.get_list()

    def load_data(self) -> list:
        '''load_data

        This method loads the data for the metrics

        Parameters:
            None
        Output:
            x: list
            y: list
        '''
        test = pd.read_csv("data_prep/UNSW/UNSW_test.csv")

        test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]

        return test.iloc[:, :-1].values, test.iloc[:, -1].values

    def get_list(self) -> list:
        '''get_list

        This method returns the list of the metrics of the model

        Parameters:
            None
        Output:
            list: list
        '''
        return [self.train_losses, self.test_losses]
