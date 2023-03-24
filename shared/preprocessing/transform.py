# ========================== Transform ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ================================================================

# ==================> Imports
import numpy as np

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# ==================> Classes
class Transform:
    def __init__(self, x: list, y: list, size: float, seed: int) -> None:
        """__init__

        This method is used to initialize the Transform class.

        Parameters:
            x: list
            y: list
            size: float
        Output:
            None
        """
        self.x = x
        self.y = y
        self.size = size
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.seed = seed

    def transform(self) -> None:
        """transform

        This method is used to transform the data.

        Output:
            tuple
        """

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=self.size,
                                                                                random_state=self.seed,
                                                                                stratify=self.y)
        # Marco with_centering a False por que si no me da una excepcion ya que no entra en memoria
        # el dataset. Solo me pasa con el dataset CIC_2018
        rc = RobustScaler(with_centering=False)
        nc = Normalizer()

        self.x_train = rc.fit_transform(self.x_train)
        self.x_test = rc.transform(self.x_test)
        
        self.x_train[np.isnan(self.x_train)] = 0
        self.x_test[np.isnan(self.x_test)] = 0

        self.x_train = nc.fit_transform(self.x_train)
        self.x_test = nc.transform(self.x_test)
