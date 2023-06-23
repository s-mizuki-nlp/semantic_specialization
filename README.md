# WIP version of "Semantic Specialization for Knowledge-based Word Sense Disambiguation"
* This repository is intended for internal use only and, as a result, is not well organized and contains experimental or unrelated code snippets.
* If you are interested in reproducing our study, please refer to the [semantic_specialization_for_wsd](https://github.com/s-mizuki-nlp/semantic_specialization_for_wsd) repository.

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
* For single trial (run), you can use `train_projection_heads.py`. Usage example can be found in `train_projection_heads.sh`.
  Also, `--help` argument shows the role of each argument. Note that the term "max pool margin task" is equivalent to the self-training objective in the paper.

* For multiple trial at once, you can use `batch_training_projection_heads.py`. Default (baseline) arguments, which is identical to the experiment setting in our paper, can be found in `experiment_settings/baseline.json`.
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
    pages = "3457--3470",
}
```
