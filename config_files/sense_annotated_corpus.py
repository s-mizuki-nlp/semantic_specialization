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
DIR_WSD_HARD_BENCHMARK = "/home/sakae/Windows/dataset/word_sense_disambiguation/wsd-hard-benchmark/wsd_hard_benchmark/"
DIR_WSD_HARD_BENCHMARK_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wsd-hard-benchmark/wsd_hard_benchmark/bert_embeddings/"

DIR_TRAINSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/"
DIR_TRAINSET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/SemCor/bert_embeddings/"
DIR_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/"
DIR_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/bert_embeddings/"
DIR_EXT_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/"
DIR_LOCAL = "/tmp/sakae/"

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
        "path": os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_wsdeval-all.hdf5"),
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
    },
    "ALLamended": {
        "path_corpus": os.path.join(DIR_WSD_HARD_BENCHMARK, "./ALLamended/ALLamended.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_WSD_HARD_BENCHMARK, "./ALLamended/ALLamended.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "The WSD hard benchmark [Maru+, ACL2022]: ALLamended subset: A revised and amended version of WSDEval-ALL."
    },
    "S10amended": {
        "path_corpus": os.path.join(DIR_WSD_HARD_BENCHMARK, "./S10amended/S10amended.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_WSD_HARD_BENCHMARK, "./S10amended/S10amended.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "The WSD hard benchmark [Maru+, ACL2022]: SemEval-2010 (S10amended): A revised and amended version of SemEval-2010."
    },
    "42D": {
        "path_corpus": os.path.join(DIR_WSD_HARD_BENCHMARK, "./42D/42D.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_WSD_HARD_BENCHMARK, "./42D/42D.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "The WSD hard benchmark [Maru+, ACL2022]: 42D: A novel challenge set for WSD, comprising difficult and out-of-domain words/senses."
    },
    "hardEN": {
        "path_corpus": os.path.join(DIR_WSD_HARD_BENCHMARK, "./hardEN/hardEN.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_WSD_HARD_BENCHMARK, "./hardEN/hardEN.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "description": "The WSD hard benchmark [Maru+, ACL2022]: hardEN: A 'hard' dataset built by including all the instances of ALLamended, SemEval-2010 and 42D that are disambiguated incorrectly by several state-of-the-art systems."
    },
    "softEN": {
        "path_corpus": os.path.join(DIR_WSD_HARD_BENCHMARK, "./softEN/softEN.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_WSD_HARD_BENCHMARK, "./softEN/softEN.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "description": "The WSD hard benchmark [Maru+, ACL2022]: softEN: Complement of the hardEN subset."
    },
    "ALLamended-bert-large-cased": {
        "path": os.path.join(DIR_WSD_HARD_BENCHMARK_EMBEDDINGS, "bert-large-cased_ALLamended.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "BERT-large-cased for ALLamended dataset."
    },
    "S10amended-bert-large-cased": {
        "path": os.path.join(DIR_WSD_HARD_BENCHMARK_EMBEDDINGS, "bert-large-cased_S10amended.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "BERT-large-cased for S10amended dataset."
    },
    "42D-bert-large-cased": {
        "path": os.path.join(DIR_WSD_HARD_BENCHMARK_EMBEDDINGS, "bert-large-cased_42D.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "BERT-large-cased for 42D dataset."
    },
    "hardEN-bert-large-cased": {
        "path": os.path.join(DIR_WSD_HARD_BENCHMARK_EMBEDDINGS, "bert-large-cased_hardEN.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased for hardEN dataset."
    },
    "softEN-bert-large-cased": {
        "path": os.path.join(DIR_WSD_HARD_BENCHMARK_EMBEDDINGS, "bert-large-cased_softEN.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased for softEN dataset."
    },
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
        "path": pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_SemCor.hdf5"),
            os.path.join(DIR_TRAINSET_EMBEDDINGS, "bert-large-cased_SemCor.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WSD SemCor corpora (excluding no-sense-annotated sentences) encoded by BERT-large-cased."
    },
    "SemCor+OMSTI": {
        "path_corpus": os.path.join(DIR_TRAINSET, "SemCor+OMSTI/semcor+omsti_fix_root.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_TRAINSET, "SemCor+OMSTI/semcor+omsti.gold.key.txt"),
        "lookup_candidate_senses": False,
        "filter_function": _no_entity_sentence_filter,
        "is_omsti_corpus": True,
        "description": "WSD SemCor+OMSTI corpora, excluding no-sense-annotated sentences.",
    },
    "SemCor+OMSTI-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_SemCor+OMSTI.hdf5"),
            os.path.join(DIR_TRAINSET_EMBEDDINGS, "bert-large-cased_SemCor+OMSTI.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WSD SemCor+OMSTI corpora (excluding no-sense-annotated sentences) encoded by BERT-large-cased."
    },
    "WordNet_Gloss_Corpus-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_WordNet_Gloss_Corpus.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_WordNet_Gloss_Corpus.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WordNet Gloss corpora encoded by BERT-large-cased."
    },
    "SREF_Sense_Corpus-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_SREF_Sense_Corpus.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_SREF_Sense_Corpus.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "Extended WordNet Gloss corpora (used in [Wang and Wang, EMNLP2020]) encoded by BERT-large-cased."
    }
}