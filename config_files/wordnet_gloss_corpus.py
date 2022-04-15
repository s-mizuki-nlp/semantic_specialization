#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

DIR_SREF_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/corpus/"
DIR_SREF_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/vectors/"

# evaluation dataset for all-words WSD task
cfg_gloss_corpus = {
    "SREF_Sense_Corpus": {
        "target_pos": ["n","v","s","r"],
        "concat_extended_examples": True,
        "lemma_surface_form_lowercase": False,
        "convert_adjective_to_adjective_satellite": True,
        "lst_path_extended_examples_corpus": [os.path.join(DIR_SREF_CORPUS, f"sentence_dict_{pos}") for pos in "n,v,a,r".split(",")],
        "description": "Word sense corpus used in SREF[Wang and Wang, EMNLP2020]. This corpus utilized a) WordNet lemmas, b) WordNet definition and examples, and c) examples collected by SREF authors."
    },
    "WordNet_Gloss_Corpus": {
        "target_pos": ["n","v","s","r"],
        "concat_extended_examples": False,
        "lemma_surface_form_lowercase": False,
        "convert_adjective_to_adjective_satellite": True,
        "lst_path_extended_examples_corpus": None,
        "description": "WordNet Gloss corpus sense corpus used in SREF[Wang and Wang, EMNLP2020]. This corpus utilized a) WordNet lemmas and b) WordNet definition and examples."
    }
}

cfg_embeddings = {
    "SREF_basic_lemma_embeddings": {
        "path": os.path.join(DIR_SREF_EMBEDDINGS, "emb_glosses_aug_gloss+examples.txt"),
        "target_pos": ["n","v","s","r"],
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "lemma_surface_form_lowercase": False,
        "description": "Basic lemma embeddings used in SREF[Wang and Wang, EMNLP2020]. This embeddings are computed using SREF Sense Corpus."
    }
}