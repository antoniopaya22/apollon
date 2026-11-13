# Datasets

Este directorio contiene todos los datasets que se han utilizado en el Trabajo de Fin de Grado y su EDA (Exploratory Data Analysis). Los datasets utilizados en el Trabajo de Fin de Grado son los siguientes:

* [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
* [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
* [CICIDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
* [CICDDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)

## Ejecución

Para ejecutar el preprocesamiento de los datasets, se pueden importar del paquete `datasets` las siguientes funciones
y variables:

* `preprocess_dataset`: Función que realiza el preprocesamiento de un dataset.
* `datasets_types`: Diccionario que contiene los datasets disponibles.

Ejemplo de uso:

```python
from datasets import preprocess_dataset
from shared.utils import load_data

# Preprocesar el dataset UNSW-NB15
df = load_data(["./data_prep/UNSW/UNSW_test.csv"])
df_preprocessed = preprocess_dataset(df, save=True, dataset_type="UNSW")
```
