from shared.utils import load_data
from datasets import preprocess_dataset, datasets_types
from intrusion_detection_systems import train_ids_model, show_model_metrics
import random
import pandas as pd


def main(seed: int, load_dataset: bool) -> None:
    random.seed(seed)

    if not load_dataset:
        # Preprocesar el dataset
        """     "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "./shared/data/CIC_2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "./shared/data/CIC_2017/Monday-WorkingHours.pcap_ISCX.csv", """
        df = load_data(
            [
                "./shared/data/CIC_2019/DrDoS_MSSQL.csv"
            ]
        )
        print("Dataset cargado")
        df_preprocessed = preprocess_dataset(
            df, save=True, dataset_type="CIC_2019", seed=seed, load=load_dataset)
        print("Dataset Preprocesado")
    else:
        df_preprocessed = preprocess_dataset(
            pd.DataFrame(), save=True, dataset_type="CIC_2019", seed=seed, load=load_dataset)
        print("Dataset Preprocesado")

    # ================> Entrenar el modelo de IDS <===============

    # Entrenar el modelo de IDS
    models = train_ids_model(x_train=df_preprocessed.x_train, y_train=df_preprocessed.y_train, x_test=df_preprocessed.x_test,
                             y_test=df_preprocessed.y_test, dataset="CIC_2019", models_type=["MLP"], save=True, seed=seed)

    # Mostrar las métricas del modelo de IDS
    # Comenta esta linea si no quieres ver las métricas, si solo quieres guardar los modelos, esta parte es la que mas tarda
    print("Modelos Guardados")
    for model in models:
        show_model_metrics(model, "SMT")


if __name__ == "__main__":
    seed = 42
    load_dataset = True
    main(seed, load_dataset)
