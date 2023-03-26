# Intrusion Detection Systems

En este directorio se encuentran los siguientes IDS:

* **Logistic Regression**
* **Decision Trees**
* **K-Nearest Neighbors**
* **Random Forest**

## Ejecución

Para ejecutar los IDS, se pueden importar del paquete `intrusion_detection_systems` las siguientes funciones:

* `train_ids_model`: Función que entrena un modelo de IDS.
* `show_model_metrics`: Función que muestra las métricas de un modelo de IDS por consola.

Ejemplo de uso:

```python
from datasets import preprocess_dataset
from shared.utils import load_data

# Preprocesar el dataset UNSW-NB15
df = load_data(["./data_prep/UNSW/UNSW_test.csv"])
df_preprocessed = preprocess_dataset(df, save=True, dataset_type="UNSW")

# ================> Entrenar el modelo de IDS <===============
from intrusion_detection_systems import train_ids_model, show_model_metrics

# Entrenar el modelo de IDS
model = train_ids_model(
    x_train=df_preprocessed.x_train,
    y_train=df_preprocessed.y_train,
    x_test=df_preprocessed.x_test,
    y_test=df_preprocessed.y_test,
    dataset="UNSW",
    model_type="RF",
    save=True
)

# Mostrar las métricas del modelo de IDS
show_model_metrics(model)
```


## Posibles mejoras

* Combinacion de Embedding y OneHot para CIC 19
* Intentar estandarizar más el proceso de prep de cada dataset
* Sacar más graficas del prep
* Reducir la carga de memoria de los datasets, ahora mismo petan la RAM. Oscar comento algo de la dataset de keras
* Tokenizar las labels de cada columna, ahora mismo cada uno hace lo que le sale de los eggs. Mayusculas, espacios ...
* Guardar los datasets preprocesados