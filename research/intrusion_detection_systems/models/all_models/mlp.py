# ========================== Random Forest Classifier =========================
#
#                   Author:  Sergio Arroni Del Riego
#
# =============================================================================

# ==================> Imports
import shared.models.model as m

from sklearn.neural_network import MLPClassifier


# ==================> Classes
class mlp(m.Model):
    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, seed: int) -> None:
        """__init__

        This method is used to initialize the MLP class.

        Parameters:
            x_train: Training data
            y_train: Training labels
            x_test: Test data
            y_test: Test labels
            dataset: Dataset name
        Output:
            None
        """
        super().__init__(x_train, y_train, x_test, y_test, dataset, seed=seed)
        self.exe()

    # Override
    def expecific_model(self) -> object:
        """expecific_model

        This method is an override of the parent method for the case of the MLP model.

        Output:
            object: MLP model
        """
        mlp = MLPClassifier(
            hidden_layer_sizes=(32),
            max_iter=200,
            verbose=False,
            random_state=self.seed,
            batch_size=200,
            early_stopping=True,
            activation='tanh',
            solver='adam'
        )
        mlp.fit(self.x_train, self.y_train)
        return mlp

    # Override
    def __str__(self) -> str:
        """__str__

        This method is used to return the name of the class.

        Output:
            str: Name of the class
        """
        return "mlp"
