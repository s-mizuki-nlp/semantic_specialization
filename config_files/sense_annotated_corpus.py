#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.filter import EmptyFilter, DictionaryFilter
_no_entity_sentence_filter = EmptyFilter(check_field_names=["entities"])
# _noun_verb_entity_selector = DictionaryFilter(includes={"pos":{"n","v"}})

from .utils import pick_first_available_path

DIR_EVALSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/Evaluation_Datasets/"
DIR_EVALSET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/bert_embeddings/"
DIR_TRAINSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/"
DIR_TRAINSET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/SemCor/bert_embeddings/"
DIR_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/"
DIR_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/bert_embeddings/"
DIR_EXT_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/"
DIR_TRAIN_UNSUPERVISED_LOCAL = "/tmp/sakae/"

# evaluation dataset for all-words WSD task
cfg_evaluation = {
    "WSDEval-ALL": {
        "path_corpus": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: ALL",
    },
    "WSDEval-ALL-concat-2": {
        "path_corpus": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "lookup_candidate_senses": True,
        "num_concat_surrounding_sentences": 2,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: Concatenates both before and after two sentences in the same document. ALL PoS tags.",
    },
    "WSDEval-ALL-bert-large-cased": {
        "path":os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_wsdeval-all.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017] encoded by BERT-large-cased."
    },
    "WSDEval-ALL-concat-2-bert-large-cased": {
        "path":os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_WSDEval-ALL-concat-2.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017] encoded by BERT-large-cased. Concatenates both before and after two sentences in the same document."
    },
    "WSDEval-ALL-mwe0.5-bert-large-cased": {
        "path":os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_WSDEval-ALL_mwe-0.5.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017] encoded by BERT-large-cased. Weighted avg between entity embeddigns and masked word embeddings."
    }
}

cfg_training = {
    "SemCor": {
        "path_corpus": os.path.join(DIR_TRAINSET, "SemCor/semcor.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_TRAINSET, "SemCor/semcor.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "description": "WSD SemCor corpora, excluding no-sense-annotated sentences.",
    },
    "SemCor-bert-large-cased": {
        "path":os.path.join(DIR_TRAINSET_EMBEDDINGS, "bert-large-cased_SemCor.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "WSD SemCor corpora (excluding no-sense-annotated sentences) encoded by BERT-large-cased."
    },
    "WordNet_Gloss_Corpus-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_TRAIN_UNSUPERVISED_LOCAL, "bert-large-cased_WordNet_Gloss_Corpus.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_WordNet_Gloss_Corpus.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WordNet Gloss corpora encoded by BERT-large-cased."
    },
    "SREF_Sense_Corpus-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_TRAIN_UNSUPERVISED_LOCAL, "bert-large-cased_SREF_Sense_Corpus.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_SREF_Sense_Corpus.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "Extended WordNet Gloss corpora (used in [Wang and Wang, EMNLP2020]) encoded by BERT-large-cased."
    }
}