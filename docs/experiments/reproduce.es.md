# Reproducir el artículo

Hay dos formas de ejecutar los experimentos: la **línea de comandos** del paquete, que cubre
el flujo completo con un comando por paso, y los **notebooks originales** de
[`research/`](https://github.com/antoniopaya22/apollon/tree/main/research), con los que se
obtuvieron los resultados del artículo.

## Con la línea de comandos

```bash
pip install -e ".[attacks]"

# 1. Data (one per dataset)
apollon prepare --dataset cic-ids2017     --raw data/CIC-IDS2017     --out data/cic17.npz
apollon prepare --dataset cse-cic-ids2018 --raw data/CSE-CIC-IDS2018 --out data/cic18.npz
apollon prepare --dataset cic-ddos2019    --raw data/CIC-DDoS2019    --out data/cic19.npz

# 2. Apollon with the pool of the paper
for d in cic17 cic18 cic19; do
  apollon train    --data data/$d.npz --out models/$d.joblib
  apollon evaluate --data data/$d.npz --model models/$d.joblib --json >> results/$d.jsonl
done

# 3. Adversarial evaluation, both attacks and settings
for d in cic17 cic18 cic19; do
  for a in zoo hsja; do
    for s in transfer direct; do
      apollon attack --data data/$d.npz --model models/$d.joblib \
                     --attack $a --setting $s --n 100 --max-iter 100 --json >> results/$d.jsonl
    done
  done
done
```

Los clasificadores individuales, que son las referencias de comparación del artículo,
equivalen a Apollon con un conjunto de un solo brazo y un solo clúster (el brazo se entrena
entonces con el 80 % de la partición de entrenamiento que no se reserva para las posteriores):

```bash
for arm in RF DT NB LR MLP; do
  apollon train --data data/cic17.npz --out models/cic17-$arm.joblib --arms $arm --clusters 1
  apollon evaluate --data data/cic17.npz --model models/cic17-$arm.joblib --json
done
```

## Con los notebooks originales

```bash
pip install -e ".[attacks]"          # the notebooks import the package
pip install -r research/requirements.txt   # optional: the exact versions of the paper
cd research
python main.py                       # preprocessing and single-classifier training
jupyter lab mab.ipynb                # Apollon on the prepared data
jupyter lab attacks/zoo.ipynb attacks/hotskipjump.ipynb   # adversarial evaluation
```

`main.py` espera los ficheros CSV de CIC-IDS2017 en `research/shared/data/CIC_2017/`;
establece `load_dataset = False` la primera vez para que los ficheros originales se
preprocesen y se guarden.

| Notebook | Contenido |
|---|---|
| `research/mab.ipynb` | Apollon (muestreo de Thompson por clúster) sobre un conjunto de datos preparado, mezcla de brazos y métricas |
| `research/attacks/zoo.ipynb` | ZOO contra los clasificadores individuales y contra Apollon |
| `research/attacks/hotskipjump.ipynb` | HopSkipJump contra los clasificadores individuales y contra Apollon |
| `research/attacks/gan.ipynb` | Ejemplos adversarios generativos |

## Hardware y tiempo

Todos los experimentos se ejecutan en CPU. En una máquina de 16 núcleos, entrenar el conjunto
con CIC-IDS2017 lleva unos minutos; CSE-CIC-IDS2018 necesita unos 32 GB de RAM sin
`--sample`. Los ataques de caja negra dominan el tiempo de ejecución: 100 muestras con 100
iteraciones tardan desde minutos (HopSkipJump en el escenario de transferencia) hasta horas
(ZOO en el escenario directo).
