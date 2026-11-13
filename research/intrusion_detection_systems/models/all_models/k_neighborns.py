# ========================== K Neighbors Classifier =========================
#
#                   Author:  Sergio Arroni Del Riego
#
# =============================================================================

# ==================> Imports
from sklearn.neighbors import KNeighborsClassifier
import shared.models.model as m


# ==================> Classes
class kNeighborns(m.Model):
    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, seed: int, k: int = 3) -> None:
        """__init__

        This method is used to initialize the kNeighborns class.

        Parameters:
            x_train: Training data
            y_train: Training labels
            x_test: Test data
            y_test: Test labels
            dataset: Dataset name
            k: Number of neighbors
        Output:
            None
        """
        self.k = k
        super().__init__(x_train=x_train, y_train=y_train, x_test=x_test,
                         y_test=y_test, dataset=dataset, seed=seed)
        self.exe()

    # Override
    def expecific_model(self) -> KNeighborsClassifier:
        """expecific_model

        This method is an override of the parent method for the case of the KNeighbors model.

        Output:
            KNeighborsClassifier
        """
        return KNeighborsClassifier(n_neighbors=self.k).fit(X=self.x_train, y=self.y_train)

    # Override
    def __str__(self) -> str:
        """__str__

        This method is used to return the name of the class.

        Output:
            str: Name of the class
        """
        return f"k_neig({self.k})"
