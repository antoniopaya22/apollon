# Salidas registradas

`research/results/` guarda las salidas en bruto de las ejecuciones de los experimentos del
artículo, tal como las imprimieron los scripts y los notebooks.

| Fichero | Contenido |
|---|---|
| `apollon_results.txt` | Entrenamiento de los clasificadores individuales con CIC-IDS2017: exactitud con validación cruzada, tasa de detección, F1 y ROC AUC en entrenamiento y prueba, matrices de confusión, informes de clasificación y tiempos de entrenamiento y predicción |
| `2018.txt` | Lo mismo para CSE-CIC-IDS2018 |
| `2019.txt` | Lo mismo para CIC-DDoS2019 |
| `MAB.txt` | Informe de clasificación de una ejecución de Apollon |
| `Apollon.xlsx` | Tablas de resultados recopiladas para el artículo |
| `arm_selections.log` | Brazo seleccionado, predicción y etiqueta real de cada petición de una ejecución de Apollon |

Las cifras definitivas, con la comparación frente a los clasificadores individuales y la
evaluación adversaria, son las del
[artículo](https://doi.org/10.1016/j.cose.2023.103546). Para regenerar los resultados con el
paquete actual, sigue [Reproducir el artículo](reproduce.md); las salidas `--json` son fáciles
de reunir en tablas.
