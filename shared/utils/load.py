# ========================== Load Data & Models Utils ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ==============================================================================

# ==================> Imports
import pandas as pd
import pickle


# ==================> Functions
def load_data(path: list, seed: int) -> pd.DataFrame:
    """load_data

        This function loads the data from the path and returns a dataframe of the datasets

        Parameters:
            path (list): list of paths to the files
        Output:
            df (pd.DataFrame): dataframe with all the data
    """
    i = 0
    df = pd.read_csv(path[i])
    while True:
        i += 1
        if i == len(path):
            break
        df = pd.concat([df, pd.read_csv(path[i])])
    return df.sample(frac=1, random_state=seed)


def load_model(name: str) -> object:
    # load the model from disk
    return pickle.load(open(name, 'rb'))
