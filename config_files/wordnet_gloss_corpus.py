#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from dataset.filter import EmptyFilter, DictionaryFilter

DIR_SREF_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/corpus/"
DIR_SREF_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/vectors/"

# evaluation dataset for all-words WSD task
cfg_gloss_corpus = {
    "SREF_Sense_Corpus": {
        "target_pos": ["n","v","a","r"],
        "concat_extended_examples": True,
        "lemma_lowercase": True,
        "lst_path_extended_examples_corpus": [os.path.join(DIR_SREF_CORPUS, f"sentence_dict_{pos}") for pos in "n,v,a,r".split(",")],
        "description": "Word sense corpus used in SREF[Wang and Wang, EMNLP2020]. This corpus utilized a) WordNet lemmas, b) WordNet definition and examples, and c) examples collected by SREF authors."
    }
}
