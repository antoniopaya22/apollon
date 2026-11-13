# Reproducing the paper

There are two ways to run the experiments: the **command line** of the package, which
covers the full pipeline with one command per step, and the **original notebooks** in
[`research/`](https://github.com/antoniopaya22/apollon/tree/main/research), which
produced the paper.

## With the command line

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

Single classifiers, the baselines of the paper, are Apollon with a pool of one arm and
one cluster (the arm is then trained on the 80 % of the training split that is not held
out for the posteriors):

```bash
for arm in RF DT NB LR MLP; do
  apollon train --data data/cic17.npz --out models/cic17-$arm.joblib --arms $arm --clusters 1
  apollon evaluate --data data/cic17.npz --model models/cic17-$arm.joblib --json
done
```

## With the original notebooks

```bash
pip install -e ".[attacks]"          # the notebooks import the package
pip install -r research/requirements.txt   # optional: the exact versions of the paper
cd research
python main.py                       # preprocessing and single-classifier training
jupyter lab mab.ipynb                # Apollon on the prepared data
jupyter lab attacks/zoo.ipynb attacks/hotskipjump.ipynb   # adversarial evaluation
```

`main.py` expects the CIC-IDS2017 CSV files under `research/shared/data/CIC_2017/`;
set `load_dataset = False` the first time so the raw files are preprocessed and saved.

| Notebook | Content |
|---|---|
| `research/mab.ipynb` | Apollon (per-cluster Thompson sampling) on a prepared dataset, arm mix and metrics |
| `research/attacks/zoo.ipynb` | ZOO against the single classifiers and against Apollon |
| `research/attacks/hotskipjump.ipynb` | HopSkipJump against the single classifiers and against Apollon |
| `research/attacks/gan.ipynb` | Generative adversarial examples |

## Hardware and time

Every experiment runs on a CPU. On a 16-core machine, training the pool on
CIC-IDS2017 takes a few minutes; CSE-CIC-IDS2018 needs about 32 GB of RAM without
`--sample`. Black-box attacks dominate the running time: 100 samples with 100
iterations take from minutes (HopSkipJump on the transfer setting) to hours (ZOO on the
direct setting).
