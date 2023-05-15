# Under construction
* This repository is under maintenance for internal use only.

# Word Sense Disambiguation
* This repostitory is the implementation of the paper "Semantic Specialization for knowledge-based Word Sense Disambiguation."

## Missing codes and objects (as of May. 2023)
* We will add following codes and objects later. These codes are only available in jupyter notebook format, so far.
* Pre-computation script of BERT embeddings for context / sense embeddings.
* ~~Independently executable evaluation script.
* Trained models (embeddings transformation functions).

# Contents

## Configuration

### `config_files/sense_annotated_corpus.py`
* Configuration file for sense-annotated corpora/dataset. Suffix `-bert-large-cased` corresponds to the pre-computed BERT embeddings.
* Evaluation dataset
  * WSDEval-ALL(-bert-large-cased): WSD Evaluation Framework dataset [Raganato+, 2017].
* Training dataset
  * SemCor(-bert-large-cased): SemCor[Miller, 1993]. Used for Self-training objective. NOTE: We discard annotated senses during training.

### `config_files/wordnet_gloss_corpus.py`
* Configuration file for WordNet Gloss Corpus and sense embeddings.
* Sense embeddings
  * SREF_basic_lemma_embeddings_without_augmentation: Sense embeddings. Specifically, this is equivalent to basic lemma embeddings used in SREF[Wang and Wang, 2020]. T
  * NOTE: This embeddings are computed without using augmented example sentences.

## Training / Evaluation
* You can use `batch_train_projection_heads.py`. Usage example can be found in `train_projection_heads.sh`.
* For multiple trial at once, you can use `batch_training_projection_heads.py`. Default (baseline) arguments can be found in `experiment_settings/baseline.json`.
* `--help` argument shows the description. The term "max pool margin task" is identical to the self-training objective in the paper.
* When finished training, it will save the trained models and evaluation results (if specified).
  * Trained models: `./checkpoints/{name}/version_{version}/checkpoints/last.ckpt`
  * Evaluation result: `{save_eval_metrics}`
 
# Reference

```
@inproceedings{Mizuki:EACL2023,
    title     = "Semantic Specialization for Knowledge-based Word Sense Disambiguation",
    author    = "Mizuki, Sakae and Okazaki, Naoaki",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    series = {EACL},
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    pages = "3449--3462",
}
```
