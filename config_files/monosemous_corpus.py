#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
from .utils import pick_first_available_path
from dataset.filter import EmptyFilter, SequenceLengthFilter
from dataset.deprecated import FrequencyBasedMonosemousEntitySampler

DIR_MONO_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/monosemous_word_annotated_corpus/"
DIR_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/monosemous_word_annotated_corpus/bert_embeddings/"
DIR_LOCAL = "/tmp/sakae/"

# We use NDJSONDataset class
# we also recommend EmptyFilter, SequenceLengthFilter, and FrequencyBasedMonosemousEntitySampler.
# EmptyFilter: remove sentence with no annotation.
# SequenceLengthFilter: limit min/max sequence length.
# FrequencyBasedMonosemousEntitySampler: limit number of occurence
cfg_corpus = {
    "monosemous_wikitext103": {
        "path": os.path.join(DIR_MONO_CORPUS, "wikitext103_train_pos-all.jsonl"),
        "path_stats": os.path.join(DIR_MONO_CORPUS, "wikitext103_train_pos-all_stats.jsonl"),
        "binary": False,
        "description": "Wikitext103 trainset > PoS tagging, lemmatization + MWE extraction > monosemous lemma sense annotation.",
    },
    "monosemous_wiki40b_all": {
        "path": os.path.join(DIR_MONO_CORPUS, "wiki40b_train_all_paragraph_pos-all.jsonl"),
        "path_stats": os.path.join(DIR_MONO_CORPUS, "wiki40b_train_all_paragraph_pos-all_stats.jsonl"),
        "binary": False,
        "description": "Wiki40b whole dataset > PoS tagging, lemmatization + MWE extraction > monosemous lemma sense annotation.",
    },
}

cfg_embeddings = {
    "monosemous_wikitext103_1-10": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_monosemous_wikitext103_freq=1-10_len=6-128_random=True.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "bert-large-cased_monosemous_wikitext103_freq=1-10_len=6-128_random=True.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wikitext-103 trainset, min_freq=1, max_sample=10, length=6~128."
    },
    "monosemous_wiki40b-all_10-10": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_monosemous_wiki40b_all_freq=10-10_len=6-128_random=True.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "bert-large-cased_monosemous_wiki40b_all_freq=10-10_len=6-128_random=True.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. wiki40b trainset(all), min_freq=10, max_sample=10, length=6~128."
    }
}
