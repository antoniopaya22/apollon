# Conjuntos de datos

Apollon funciona con cualquier tabla de características de flujos con una etiqueta binaria.
El paquete incluye cargadores para los tres conjuntos de datos del Canadian Institute for
Cybersecurity utilizados en el artículo. Los conjuntos de datos no se redistribuyen:
descárgalos desde sus páginas oficiales.

| Nombre (`--dataset`) | Conjunto de datos | Ficheros que descargar | Columna de etiqueta |
|---|---|---|---|
| `cic-ids2017` | [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | `MachineLearningCSV.zip` (8 ficheros CSV) | ` Label` (`BENIGN` / nombre del ataque) |
| `cse-cic-ids2018` | [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | `Processed Traffic Data for ML Algorithms` (10 ficheros CSV, del bucket de AWS) | `Label` (`Benign` / nombre del ataque) |
| `cic-ddos2019` | [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) | `CSV-01-12.zip` y/o `CSV-03-11.zip` | ` Label` (`BENIGN` / nombre del ataque) |

!!! info "Nombres de columna"
    Los ficheros CIC conservan un espacio inicial en muchos nombres de columna (`" Label"`,
    `" Destination Port"`). Apollon mantiene los nombres originales para que la lista de
    características funcionales de la [evaluación adversaria](attacks.md) coincida con ellos.

## Preprocesamiento

<figure class="ap-fig">
<svg viewBox="0 0 720 120" role="img" aria-label="Flujo de preprocesamiento: ficheros CSV, limpieza, etiquetas, partición, escalado, npz.">
<defs><marker id="ap-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8e95c9"/></marker><marker id="ap-arr-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f43f5e"/></marker></defs>
<rect class="box-hi" x="10" y="30" width="100" height="64" rx="12"/><text class="t" x="60" y="58" text-anchor="middle">Ficheros CSV</text><text class="m" x="60" y="78" text-anchor="middle">CIC en bruto</text><path class="arrow flow" d="M112,62 H128"/><rect class="box" x="130" y="30" width="100" height="64" rx="12"/><text class="t" x="180" y="58" text-anchor="middle">Limpieza</text><text class="m" x="180" y="78" text-anchor="middle">sin ids/constantes</text><path class="arrow flow" d="M232,62 H248"/><rect class="box" x="250" y="30" width="100" height="64" rx="12"/><text class="t" x="300" y="58" text-anchor="middle">Etiquetas</text><text class="m" x="300" y="78" text-anchor="middle">0 benigno · 1 ataque</text><path class="arrow flow" d="M352,62 H368"/><rect class="box" x="370" y="30" width="100" height="64" rx="12"/><text class="t" x="420" y="58" text-anchor="middle">Partición</text><text class="m" x="420" y="78" text-anchor="middle">70/30 por etiqueta</text><path class="arrow flow" d="M472,62 H488"/><rect class="box" x="490" y="30" width="100" height="64" rx="12"/><text class="t" x="540" y="58" text-anchor="middle">Escalado</text><text class="m" x="540" y="78" text-anchor="middle">robusto + norma 1</text><path class="arrow flow" d="M592,62 H608"/><rect class="box-hi" x="610" y="30" width="100" height="64" rx="12"/><text class="t" x="660" y="58" text-anchor="middle">.npz</text><text class="m" x="660" y="78" text-anchor="middle">fichero portable</text>
</svg>
<figcaption><code>apollon prepare</code>: de los ficheros CSV originales de un conjunto de datos a una partición de entrenamiento/prueba portable. Los escaladores se ajustan solo con la partición de entrenamiento.</figcaption>
</figure>


`apollon prepare` (o [`apollon.data.prepare`](../reference/data.md)) ejecuta el
flujo de preprocesamiento del artículo:

1. lee y concatena los ficheros CSV (un directorio se expande a sus ficheros `*.csv`) y
   los baraja con una semilla fija;
2. elimina las columnas identificadoras (id del flujo, direcciones, puertos y marcas de
   tiempo, cuando el conjunto de datos las tiene), las líneas de cabecera repetidas dentro de
   los ficheros y las características constantes;
3. sustituye los valores infinitos y ausentes por 0 y convierte las etiquetas en 0 (benigno)
   o 1 (cualquier ataque);
4. divide los datos 70/30 en entrenamiento y prueba, de forma estratificada por etiqueta;
5. escala con `RobustScaler(with_centering=False)` y `Normalizer`, ambos ajustados solo con la
   partición de entrenamiento.

```bash
apollon prepare --dataset cse-cic-ids2018 --raw data/CSE-CIC-IDS2018 \
                --out data/cic18.npz --sample 500000
```

La salida es un único fichero `.npz` comprimido con la partición de entrenamiento/prueba, los
nombres de las características y el nombre del conjunto de datos, que se carga con
`apollon.data.Dataset.load`.

!!! tip "Memoria"
    CSE-CIC-IDS2018 tiene unos 16 millones de flujos. Usa `--sample` para conservar un
    subconjunto estratificado, o prepáralo en una máquina con al menos 32 GB de RAM.

## Tus propios datos

Cualquier tabla de `pandas` sirve a través de las funciones de más bajo nivel:

```python
import pandas as pd
from apollon.data import DatasetSpec, clean, split_and_scale

spec = DatasetSpec(name="My network", label="label", benign="normal",
                   drop=("src_ip", "dst_ip", "timestamp"))
X, y = clean(pd.read_csv("flows.csv"), spec)
data = split_and_scale(X, y, seed=0)
data.save("data/my-network.npz")       # usable by `apollon train`
```
