# SEMKR: Joint Learning of Semantic and Topological Representations for Knowledge Graph Completion

This repository contains the implementation of SEMKR, a model designed for Knowledge Graph Completion (KGC) with semantic-enhanced graph reasoning.

## Datasets

The current release supports the following 4 general domain datasets:
- **FB15k-237**
- **WN18**
- **WN18RR**
- **NELL995**

Note: The `LegalPP` dataset is currently excluded from this version.

## Usage

### 1. Pre-training (EDRL)

Before running the main model, perform semantic pre-training using EDRL. Ensure you have a local BERT model (e.g., `bert-base-uncased`).

```bash
python3 edrl.py \
    --dataset [DATASET_NAME] \
    --bert_model /path/to/bert-base-uncased \
    --kg_dim 400 \
    --temperature 0.05 \
    --n_neg 10
```

Supported `[DATASET_NAME]`: `FB15k-237`, `WN18`, `WN18RR`, `NELL995`.

### 2. Feature Preparation

Link the generated semantic features to the data directory:

```bash
ln -sf /path/to/MUSE/data/[DATASET_NAME]/entity_features_pretrained.npy /path/to/MUSE/data/[DATASET_NAME]/bert.npy
```

### 3. Main Model Training

Run the main reasoning model:

```bash
python3 main.py \
    --dataset [DATASET_NAME] \
    --dim 400 \
    --feature_type bert \
    --cuda
```

## Results

Performance on general domain benchmarks:

| Dataset | MRR | Hits@1 | Hits@3 |
| :--- | :--- | :--- | :--- |
| FB15k-237 | 0.986 | 0.977 | 0.996 |
| WN18 | 0.998 | 0.996 | 1.000 |
| WN18RR | 0.993 | 0.985 | 1.000 |
| NELL995 | 0.953 | 0.925 | 0.978 |

---
*For internal research and evaluation purposes.*
