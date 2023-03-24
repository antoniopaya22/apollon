from shared.utils import load_data
from datasets import preprocess_dataset, datasets_types
from intrusion_detection_systems import train_ids_model, show_model_metrics
import random

seed = 42

random.seed(seed)

# Preprocesar el dataset UNSW-NB15
df = load_data(
    ["./shared/data/CIC_2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "./shared/data/CIC_2017/Wednesday-workingHours.pcap_ISCX.csv"] )

print("Dataset cargado")

df_preprocessed = preprocess_dataset(
    df, save=True, dataset_type="CIC_2017", seed=seed)

print("Dataset Preprocesado")

# ================> Entrenar el modelo de IDS <===============

# Entrenar el modelo de IDS
models = train_ids_model(x_train=df_preprocessed.x_train, y_train=df_preprocessed.y_train, x_test=df_preprocessed.x_test,
                        y_test=df_preprocessed.y_test, dataset="CIC_2017", models_type=["LR", "SVC", "MLP", "KNN", "DT", "RF"], save=True, seed=seed)

# Mostrar las métricas del modelo de IDS
# Comenta esta linea si no quieres ver las métricas, si solo quieres guardar los modelos, esta parte es la que mas tarda,
# tambien puedes cambiar n_split en la linea 23 de show_metrics_test.py, ahora esta en 5 que esta bien para la literatura, pero lo dicho tarda mucho.
print("Modelos Guardados")
for model in models:
    show_model_metrics(model, "SMT")
