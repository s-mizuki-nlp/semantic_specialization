#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

DIR_SREF_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/corpus/"
DIR_SREF_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/SREF/vectors/"

from .sense_annotated_corpus import cfg_training

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
    },
    "SREF_enhanced_embeddings": {
        "path": os.path.join(DIR_SREF_EMBEDDINGS, "emb_wn_all-relations_aug_gloss+examples.pkl"),
        "target_pos": ["n","v","s","r"],
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "lemma_surface_form_lowercase": False,
        "description": "Semantic Relation Enhanced lemma embeddings used in SREF[Wang and Wang, EMNLP2020]. This embeddings are the post-processing of basic lemma embeddings."
    },
    "SREF_basic_lemma_embeddings_without_augmentation": {
        "path": os.path.join(DIR_SREF_EMBEDDINGS, "emb_glosses_gloss+examples.txt"),
        "target_pos": ["n","v","s","r"],
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "lemma_surface_form_lowercase": False,
        "description": "Basic lemma embeddings used in SREF[Wang and Wang, EMNLP2020]. This embeddings are computed without using augmented example sentences."
    },
    "SREF_Sense_Corpus-ANY-bert-large-cased": {
        "kwargs_bert_embeddings_dataset": cfg_training["SREF_Sense_Corpus-bert-large-cased"],
        "pooling_method": None,
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "description": "SREF Extended WordNet Gloss sentence embeddings."
    },
    "WordNet_Gloss_Corpus-ANY-bert-large-cased": {
        "kwargs_bert_embeddings_dataset": cfg_training["WordNet_Gloss_Corpus-bert-large-cased"],
        "pooling_method": "average",
        "l2_norm": False,
        "use_first_embeddings_only": True,
        "description": "WordNet Gloss sentence embeddings."
    }
}