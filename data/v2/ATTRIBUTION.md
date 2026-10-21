# Data attribution (dataset v2)

## CLINC150

Part of dataset v2 is derived from **CLINC150** ("plus" configuration), retrieved from
https://huggingface.co/datasets/clinc/clinc_oos at revision `155b9c710419136e17307b80d0a13e68cd46b4ec`.

> Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill,
> Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019.
> *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* In Proceedings of
> EMNLP-IJCNLP 2019. https://www.aclweb.org/anthology/D19-1131

License: [Creative Commons Attribution 3.0 (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/).

**Changes made** (`src/data_v2.py`): CLINC intents were relabeled to this project's intents
(`thank_you` → `thanks`, `goodbye` → `goodbye`, `what_can_i_ask_you` and `are_you_a_bot` → `capabilities`,
`oos` and one example from each non-medical intent → `out_of_scope`). CLINC's `greeting` was split into
`greeting` and `how_are_you` by a wording rule. Rows were subsampled, and rows whose text conflicted with
another source were dropped. CLINC's own train/validation/test assignment was kept. Rows from CLINC are
marked `"source": "clinc"` in `data/v2/splits/*.jsonl`.

## Other sources

- `v1`: `data/intents.json`, the project's original patterns.
- `authored`: `data/v2/authored.json`, patterns written for this project in October 2026.
