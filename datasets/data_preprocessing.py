# ========================== Data Preprocessing ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ======================================================================

# ==================> Imports
import pandas as pd

from shared.preprocessing import CIC_2017, CIC_2018, CIC_2019, transform
from shared.preprocessing.clear_data import ClearData

# ===================> Enumerations
datasets_types = {
    "CIC_2017": CIC_2017,
    "CIC_2018": CIC_2018,
    "CIC_2019": CIC_2019
}


# ==================> Functions
def preprocess_dataset(df: pd.DataFrame, save: bool, dataset_type: str, seed: int, name_save: str, name_load: str, load: bool = True) -> transform:
    """preprocess_dataset

    This function preprocesses a dataset

    Parameters:
        df: Dataframe to preprocess
        save: Save the dataset
        dataset_type: Type of dataset
    Output:
        Transform object
    """
    data: ClearData = datasets_types[dataset_type](
        df=df, do_save=save, seed=seed, name_save=name_save, name_load=name_load)
    if load:
        print("Loading existing data")
        x, y = data.load_data()
        trans = transform(x=x, y=y, size=0.3, seed=seed)
    else:
        print("Loading new data")
        data.clear_data()
        trans = transform(x=data.x, y=data.y, size=0.3, seed=seed)
    trans.transform()
    return trans
