# ========================== Save Data & Models Utils ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ==============================================================================

# ==================> Imports
import pickle
import pandas as pd


def save_model(model, name: str) -> None:
    """save_model

        This function saves the model

        Parameters:
            model: model to save
            name (str): name of the model
        Output:
            None
    """
    # Its important to use binary mode
    name = f"intrusion_detection_systems/models/saved_models/{name}"
    save = open(name, 'wb')

    # source, destination
    pickle.dump(model, save)

    # close the file
    save.close()

def save_data(df: pd.DataFrame, name: str) -> None:
    """save_data

        This function saves the data

        Parameters:
            df (pd.DataFrame): dataframe to save
            name (str): name of the dataframe
        Output:
            None
    """
    df.to_csv(f"./data_prep/merged/{name}.csv", index=False)