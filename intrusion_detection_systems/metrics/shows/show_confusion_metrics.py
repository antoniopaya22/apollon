import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import intrusion_detection_systems.metrics.show_metrics as s_m


class ConfusionMatrix(s_m.ShowMetrics):

    # Override
    def operation(self) -> None:
        '''operation

        This method represents the specific behavior that this class has,
        it prints the confusion matrix for the predictions made with the test value.

        Parameters:
            None
        Output:
            None
        '''
        plt.rcParams['figure.figsize'] = 5, 5
        sns.set_style("white")
        ConfusionMatrixDisplay.from_predictions(
            self._component.y_test, self._component.predictions, cmap=plt.cm.Blues)
        plt.show()
