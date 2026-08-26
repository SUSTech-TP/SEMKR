# Data preparation and expected file layout

## Scope of the public snapshot

This repository does not redistribute raw benchmark triples, entity dictionaries, complete entity descriptions, pretrained checkpoints, or legal-domain datasets. The `data/` directory currently contains only relation-to-text mapping files for four general-domain datasets. These files are retained as lightweight research artifacts, but they are **not located at the paths read directly by `edrl.py`**. Users must prepare the complete directory structure below before running the EDRL pipeline.

Please obtain every benchmark from its original provider and comply with its licence, terms of use, and citation requirements. The repository maintainers make no claim to redistribute these third-party datasets.

## Required input files for EDRL

For a supported dataset identifier, the EDRL script requires the following files.

| Dataset identifier passed to `--dataset` | Triples and dictionaries | Text-description files | Built-in fallback |
|---|---|---|---|
| `FB15k-237` | `data/FB15k-237/` | `data_with_text/FB15k-237/` | Entity/relation names are used if no text file is present. |
| `wn18` | `data/wn18/` | `data_with_text/WN18/` | Entity/relation names are used if no text file is present. |
| `wn18rr` | `data/wn18rr/` | `data_with_text/WN18RR/` | Entity/relation names are used if no text file is present. |
| `NELL995` | `data/NELL995/` | Not required by the current configuration | Entity/relation names are used. |
| `LegalPP` | `data/LegalPP/` | `data_with_text/LegalPP/` | Entity/relation names are used if no text file is present. |
| `LegalPP_link` | `data/LegalPP_link/` | `data_with_text/LegalPP_link/` | Entity/relation names are used if no text file is present. |

For a dataset rooted at `data/<dataset>/`, EDRL reads the following files:

```text
data/<dataset>/
├── entities.dict                 # required
├── relations.dict                # required
└── train.txt                     # required
```

The supported formats are:

```text
# entities.dict and relations.dict
<integer_id>\t<name>

# train.txt
<head_name>\t<relation_name>\t<tail_name>
```

Names used in `train.txt` must match the names in the corresponding dictionary files. `entities.dict` is assumed to contain contiguous integer identifiers beginning at 0.

## Optional text descriptions

Text-description files enrich EDRL inputs. Where configured, each file is tab-separated:

```text
<entity_or_relation_identifier>\t<free-text description>
```

For `FB15k-237`, `WN18`, and `WN18RR`, the first field should be the identifier expected by the script's `tab_id_text` configuration. For `LegalPP` and `LegalPP_link`, it should be the entity or relation name expected by the `tab_name_text` configuration. When a description cannot be found, EDRL falls back to the name with underscores and slashes normalized to spaces.

## Existing mapping files

The tracked mapping files are located at:

```text
data/FB15k-237/relation2text.txt
data/WN18/relation2text.txt
data/WN18RR/relation2text.txt
data/NELL995/relation2text.txt
data/general/relation2text.txt
```

They do not by themselves create an executable EDRL dataset. Before an end-to-end release, users must reconcile their content and identifiers with the `data_with_text/<dataset>/relation2text.txt` locations expected by `edrl.py`. This document intentionally records the difference rather than silently relocating or transforming the files.

## Generated artifacts

A successful EDRL run writes the following artifacts into `data/<dataset>/`:

```text
entity_features_pretrained.npy
relation_features_pretrained.npy
```

It also writes model checkpoints to the repository root:

```text
edrl_pretrained_<dataset>.pt
edrl_finetuned_<dataset>.pt
```

The public repository does not currently include these generated artifacts.

## CAIL2018 sourcing and licence position

CAIL2018 is obtained separately from the official challenge release and is not redistributed with this repository. Please comply with the original dataset terms and cite the required CAIL2018 publications. The official CAIL2018 GitHub repository is distributed under the MIT License; users should not infer that this automatically grants a licence to redistribute the separately hosted raw dataset.

When using CAIL2018, cite both *CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction* and *Overview of CAIL2018: Legal Judgment Prediction Competition*, as requested in the official release documentation.
