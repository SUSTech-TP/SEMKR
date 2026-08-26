# Reproducibility statement

## Summary

SEMKR is associated with the published paper “SEMKR: Joint Learning of Semantic and Topological Representations for Knowledge Graph Completion” in *Neurocomputing* (2025). This document states precisely what can and cannot be reproduced from the current public repository snapshot.

| Component | Public source present | Required external data present | Runnable from a fresh clone | Status |
|---|---:|---:|---:|---|
| EDRL model definition and feature extraction | Yes | No | After users prepare data and a local BERT checkpoint | Documented, conditionally runnable |
| Relation-to-text mappings for four general-domain datasets | Yes | Not applicable | Yes, as reference files | Available |
| End-to-end main reasoning training | Interface only | No | No | Not yet publicly reproducible |
| Complete evaluation pipeline and raw logs | No | No | No | Not released |
| Legal-domain data and checkpoints | No | No | No | Not released |

## What the EDRL script provides

[`edrl.py`](../edrl.py) provides a self-contained implementation of the semantic feature-preparation stage. Given correctly formatted entity/relation dictionaries, training triples, optional descriptions, and a local BERT checkpoint, it performs the following stages:

1. Contrastive continued pre-training with an InfoNCE-style objective.
2. Relation-classification fine-tuning.
3. Feature extraction into `entity_features_pretrained.npy` and `relation_features_pretrained.npy`.

The script exposes a fixed random seed through `--seed` (default: `42`) and uses `torch.manual_seed`, `torch.cuda.manual_seed_all`, NumPy, and Python's `random` module. GPU execution is selected automatically when CUDA is available; `--fp16` enables automatic mixed precision.

## Preflight check

After installing the dependencies in [`requirements.txt`](../requirements.txt), verify that the included command interface loads correctly:

```bash
python edrl.py --help
```

Then prepare the external files specified in [`DATA.md`](./DATA.md) and run a dataset-specific command, for example:

```bash
python edrl.py \
  --dataset FB15k-237 \
  --bert_model /absolute/path/to/bert-base-uncased \
  --kg_dim 400 \
  --pt_epochs 5 \
  --ft_epochs 5 \
  --batch_size 32 \
  --temperature 0.05 \
  --n_neg 10 \
  --seed 42
```

A successful run should produce two checkpoint files at the repository root and two NumPy feature files within the dataset directory. Record the Python version, PyTorch version, Transformers version, hardware, command, and random seed with every run.

## Current limitations

The command interface in [`main.py`](../main.py) imports `data_loader` and `train`, but the current public repository does not contain `data_loader.py` or `train.py`. It follows that a fresh clone cannot execute the main reasoning stage or independently regenerate the benchmark table in the README. No substitute implementation, synthetic result, or hidden configuration is provided by this documentation update.

The repository also does not include original benchmark triples, complete text-description files, legal-domain data, pretrained checkpoints, evaluation scripts, or raw experiment logs. These omissions prevent full independent replication of the reported paper results. The public scope should therefore be understood as a documented research-code snapshot, not a complete archival release.

## Reporting reproducibility issues

When opening an issue, please include the following information in text form:

| Field | Example |
|---|---|
| Operating system and Python | Ubuntu 22.04; Python 3.10.13 |
| Hardware | NVIDIA A100 40 GB or CPU-only |
| Dependency versions | `torch`, `transformers`, `numpy`, `rank-bm25` |
| Dataset identifier | `FB15k-237` |
| BERT checkpoint | Exact local model name or revision |
| Command | Complete shell command without private paths or tokens |
| Random seed | `42` |
| Failure information | Full traceback and a minimal description of prepared data files |

This information enables maintainers and users to distinguish environment issues from data-format and implementation issues.
