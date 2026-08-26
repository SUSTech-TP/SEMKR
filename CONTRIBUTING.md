# Contributing to SEMKR

Thank you for your interest in SEMKR. This repository is maintained as a research-code release associated with the SEMKR paper. Contributions that improve documentation accuracy, reproducibility, data-format clarity, and maintainability are welcome.

## Before opening an issue

Please first read the [reproducibility statement](./docs/REPRODUCIBILITY.md) and the [data-format specification](./docs/DATA.md). The current public snapshot does not contain the complete end-to-end main reasoning pipeline or raw benchmark data. Reports caused by absent `data_loader.py`, `train.py`, raw data, full text-description files, legal-domain data, or unpublished checkpoints should therefore be identified as release-scope limitations rather than unexpected runtime regressions.

## Reporting a bug or documentation issue

Use a descriptive title and include a minimal, non-sensitive reproduction record. The following information is particularly useful:

| Information | Why it is needed |
|---|---|
| Operating system and Python version | Helps reproduce environment-specific behaviour. |
| Hardware and CUDA availability | Distinguishes GPU, CPU, and driver issues. |
| Package versions | Identifies dependency incompatibilities. |
| Dataset identifier and file layout | Detects format and path mismatches. |
| Full command and random seed | Enables deterministic investigation where possible. |
| Complete traceback or proposed wording change | Provides an actionable starting point. |

Do not upload proprietary data, private case records, credentials, model-access tokens, or any personally identifiable information.

## Pull requests

Before proposing a pull request, please keep the change focused, explain its relationship to the paper or released code, and avoid modifying reported benchmark values without reproducible evidence. Documentation changes should preserve the distinction between what is publicly released, what is described in the paper, and what remains unavailable in this snapshot.

All contributors are expected to respect the licences and access conditions of third-party datasets and pretrained models.
