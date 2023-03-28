# ========================== IDS Models Runner =========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ======================================================================

# ==================> Imports
from intrusion_detection_systems.models import k_neig_model, dec_tree_model, r_forest_model, log_reg_model, mlp_model, svc_model, nb_model
from intrusion_detection_systems.metrics import CM, SMT, SMLM
from shared.utils import save_model
from typing import Any

# ==================> Enumerations
models_types_default = {
    "KNN": k_neig_model,
    "DT": dec_tree_model,
    "RF": r_forest_model,
    "LR": log_reg_model,
    "MLP": mlp_model,
    "SVC": svc_model,
    "NB": nb_model
}

metrics_types = {"CM": CM, "SMT": SMT, "SMLM": SMLM}

# ==================> Functions


def train_ids_model(x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, models_type: list,
                    save: bool, seed: int) -> Any:
    """train_ids_model

    This function train a IDS model with the given parameters and saves it in the models folder

    Parameters:
        x_train: Training data
        y_train: Training labels
        x_test: Test data
        y_test: Test labels
        dataset: Dataset name
        models_type: Models type
        save: Save model
    Output:
        None
    """
    models = []
    for model in models_type:
        model_t = models_types_default[model](
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, dataset=dataset, seed=seed)
        if save:
            save_model(model_t, dataset + "_" + model)
            print(f"Modelo {model} guardado")
        models.append(model_t)
    return models


def show_model_metrics(model: Any, metric_type: str) -> None:
    """show_model_metrics

    This function shows the metrics of the given model in the console

    Parameters:
        model: Model
    Output:
        None
    """
    metrics_types[metric_type](model).operation()
    '''
    CM(model).operation()
    SMT(model).operation()
    SMLM(model).operation()
    '''
