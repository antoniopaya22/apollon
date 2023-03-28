# ========================== Clear Data - CIC IDS 2019 ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ===============================================================================

import pandas as pd

# ==================> Imports
import shared.preprocessing.clear_data as cd


# ==================> Classes
class ClearDataCIC2019(cd.ClearData):
    def __init__(self, df: pd.DataFrame, do_save: bool, seed:int) -> None:
        """__init__

        This method is used to initialize the ClearDataCIC2019 class.

        Parameters:
            df: pd.DataFrame
            do_save: bool
        Output:
            None
        """
        super().__init__(df=df, seed=seed)
        self.do_save = do_save

    # Override
    def clear_data(self) -> None:
        """clear_data

        This method is used to clear the data of the CIC 2019 dataset.

        Output:
            None
        """

        list_drop = ['Flow ID', ' Source IP', ' Source Port',
                     ' Destination IP', ' Destination Port', ' Timestamp', ' Inbound', 'SimillarHTTP', ' Protocol']
        self.df.drop(list_drop, axis=1, inplace=True)
        self.reduce_tam()
        self.best_features_func()

        self.x = self.df.drop([" Label"], axis=1)
        self.y = self.df[" Label"]

        self.replace(list_B_columns=["BENIGN"], list_M_columns=["DrDoS_MSSQL"])
        self.drop_bad_elements_x()

        '''
        # Peta por memoria
        best_features = SelectKBest(score_func=chi2, k='all')
        self.show_img_features(best_features.fit(self.x, self.y))
        [0, 1, 2, 3, 4]
        '''
        if self.do_save:
            self.save_data()

    # Override
    def save_data(self):

        aux_df = self.df
        aux_df.drop([" Label"], axis=1, inplace=True)

        aux_y = pd.DataFrame(self.y, columns=[' Label'])
        aux_df = pd.concat([aux_df, aux_y], axis=1)

        aux_df.to_csv('./shared/data_prep/CIC19/CIC19.csv', index=False)
        aux_y.to_csv('./shared/data_prep/CIC19/CIC19_y.csv', index=False)
