# SEMKR

**SEMKR: Joint Learning of Semantic and Topological Representations for Knowledge Graph Completion**

This repository hosts the research implementation accompanying the SEMKR paper, published in *Neurocomputing*. SEMKR is a text-guided knowledge graph completion framework that jointly learns semantic representations from entity descriptions and topological representations from relational context and paths. The method combines **Entity Description Representation Learning (EDRL)** and **Relational Topological Representation Learning (RTRL)** to improve relation and link prediction under heterogeneous graph-density conditions.

| Resource | Link |
|---|---|
| Published paper | [Neurocomputing, Volume 653, Article 130909](https://doi.org/10.1016/j.neucom.2025.130909) |
| Preprint | [SSRN: 5253194](https://ssrn.com/abstract=5253194) |
| Citation metadata | [`CITATION.cff`](./CITATION.cff) |
| Reproducibility notes | [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md) |
| Dataset-format specification | [`docs/DATA.md`](./docs/DATA.md) |
| Repository evolution | [`CHANGELOG.md`](./CHANGELOG.md) |

> **Repository status.** This public snapshot includes the EDRL feature-preparation implementation and relation-to-text mappings for the four general-domain benchmarks. It does **not** currently distribute raw benchmark data, full text-description files, pretrained checkpoints, or the local modules required by `main.py` (`data_loader.py` and `train.py`). The main-model command should therefore be regarded as a documented interface rather than a fully executable public release until the missing components are released. This limitation is stated explicitly to keep the repository scientifically transparent.

## 1. Method overview

SEMKR addresses knowledge graph completion (KGC) by integrating complementary semantic and topological evidence. **EDRL** continually pre-trains and fine-tunes a BERT encoder on entity and relation descriptions, then exports fixed-dimensional entity and relation features. **RTRL** incorporates local edge-aware message passing and relational-path aggregation into the KGC reasoning process. The reported paper evaluates the approach on four general-domain benchmarks and legal-domain settings.

For an implementation-level description, see the docstrings and command-line help in [`edrl.py`](./edrl.py). For the research methodology, experimental protocol, and complete results, please consult the published paper.

## 2. Repository layout

```text
SEMKR/
├── data/                         # Relation-to-text mappings included in this snapshot
│   ├── FB15k-237/
│   ├── NELL995/
│   ├── WN18/
│   └── WN18RR/
├── docs/
│   ├── DATA.md                   # Required external data layout and identifiers
│   └── REPRODUCIBILITY.md        # Scope, commands, and current limitations
├── edrl.py                       # Entity Description Representation Learning (EDRL)
├── main.py                       # Main-model interface; dependent modules are not released here
├── requirements.txt              # Python dependencies for the included EDRL pipeline
├── CITATION.cff                  # Machine-readable software and paper citation metadata
└── CHANGELOG.md                  # Transparent repository evolution record
```

## 3. Supported datasets and public scope

| Dataset | Task context | Mapping file included | Full raw data / text descriptions included |
|---|---|---:|---:|
| FB15k-237 | General-domain KGC | Yes | No |
| WN18 | General-domain KGC | Yes | No |
| WN18RR | General-domain KGC | Yes | No |
| NELL995 | General-domain KGC | Yes | No |
| LegalPP | Legal-domain KGC | No | No |
| LegalPP_link | Legal-domain KGC | No | No |

The `data/` directory intentionally contains only lightweight relation-to-text mapping files. Users must obtain the original benchmark data under their respective licences and construct the files required by the scripts. The expected layout and file format are documented in [`docs/DATA.md`](./docs/DATA.md).

## 4. Environment

The included EDRL script has been written for Python 3 and uses PyTorch, Hugging Face Transformers, NumPy, and `rank-bm25`. Create an isolated environment and install the declared dependencies:

```bash
git clone https://github.com/SUSTech-TP/SEMKR.git
cd SEMKR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A local BERT checkpoint compatible with `BertTokenizer` and `BertModel` is required. The paper uses a BERT-based encoder; for the general-domain example below, a local copy of `bert-base-uncased` may be supplied through `--bert_model`.

## 5. EDRL feature preparation

After preparing the required dataset files, run EDRL as follows:

```bash
python edrl.py \
  --dataset FB15k-237 \
  --bert_model /absolute/path/to/bert-base-uncased \
  --kg_dim 400 \
  --temperature 0.05 \
  --n_neg 10 \
  --seed 42
```

The script performs contrastive pre-training, relation-classification fine-tuning, and feature extraction. It writes the following artifacts into the selected dataset directory:

```text
edrl_pretrained_<dataset>.pt
edrl_finetuned_<dataset>.pt
entity_features_pretrained.npy
relation_features_pretrained.npy
```

The documented EDRL defaults are 5 contrastive-pre-training epochs, 5 fine-tuning epochs, a batch size of 32, a temperature of 0.05, and 10 negative relation samples. Refer to `python edrl.py --help` for the complete argument list.

## 6. Main-model interface and current release limitation

The public interface for the main reasoning stage is retained below for traceability:

```bash
python main.py \
  --dataset FB15k-237 \
  --dim 400 \
  --feature_type bert \
  --cuda
```

However, `main.py` imports local modules that are not present in the current public snapshot. As a result, this command cannot be executed from a fresh clone at present. The repository does not claim end-to-end reproducibility until those components, their configuration files, and the complete data-processing path are released. Please see [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md) before using the repository for comparison or replication.

## 7. Reported general-domain benchmark results

The following table records the general-domain results currently reported by this repository. These values should be cited together with the paper; they are not automatically regenerated by the incomplete public snapshot described above.

| Dataset | MRR | Hits@1 | Hits@3 |
|---|---:|---:|---:|
| FB15k-237 | 0.986 | 0.977 | 0.996 |
| WN18 | 0.998 | 0.996 | 1.000 |
| WN18RR | 0.993 | 0.985 | 1.000 |
| NELL995 | 0.953 | 0.925 | 0.978 |

## 8. Reproducibility and responsible use

Please obtain and use all benchmark data in accordance with their original terms. The legal-domain resources referenced in the paper are not included in this repository. Users are responsible for data governance, access permissions, privacy protection, and the appropriate interpretation of prediction outputs.

If you identify an issue in the documentation, command interface, or released source code, please open a GitHub issue with the operating system, Python version, dependency versions, dataset identifier, command, and complete error trace.

## 9. Citation

If you use SEMKR, please cite the published paper:

```bibtex
@article{liu2025semkr,
  title   = {SEMKR: Joint Learning of Semantic and Topological Representations for Knowledge Graph Completion},
  author  = {Liu, Pengjie and Zhang, Wang and Ding, Yulong and Jiang, Jie and Yang, Shuang-Hua},
  journal = {Neurocomputing},
  volume  = {653},
  pages   = {130909},
  year    = {2025},
  doi     = {10.1016/j.neucom.2025.130909},
  url     = {https://doi.org/10.1016/j.neucom.2025.130909}
}
```

## 10. Licence

The SEMKR source code and repository documentation are available under the [MIT License](./LICENSE). Third-party datasets, pretrained models, and externally hosted resources are **not** relicensed by this repository. In particular, CAIL2018 must be obtained separately from the official challenge release and is not redistributed here; see [`docs/DATA.md`](./docs/DATA.md) for the applicable sourcing, citation, and redistribution boundary.

## 11. Project evolution

The repository retains its original Git history. It was initialized on 19 March 2025, followed by the public general-domain implementation and README updates on 30–31 July 2026. Documentation updates are added as new commits only; prior commit identities and timestamps are never rewritten. See [`CHANGELOG.md`](./CHANGELOG.md) for a transparent summary.
