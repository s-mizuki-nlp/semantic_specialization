#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
from .utils import pick_first_available_path
from dataset.filter import EmptyFilter, SequenceLengthFilter
from dataset.transform import FrequencyBasedMonosemousEntitySampler

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
    }
}

cfg_embeddings = {
    "wikitext103-subset": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_wikitext103_train_freq=10-11_len=6-128.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "bert-large-cased_wikitext103_train_freq=10-11_len=6-128.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. subset of the Wikitext-103 trainset, freq=10~11, length=6~128."
    },
    "wiki40b-first-paragraph-bert-base": {
        "path":os.path.join(DIR_EMBEDDINGS, "bert-base-cased_wiki40b-train-first-paragraph_freq=10-100_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-base-cased. Wiki40b trainset first paragraph, freq=10~100, length=6~128."
    },
    "wiki40b-all": {
        "path":os.path.join(DIR_EMBEDDINGS, "bert-large-cased_wiki40b-train-all-paragraph_freq=10-100_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, freq=10~100, length=6~128."
    },
    "wiki40b-all-wide-vocab": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_wiki40b-train-all-paragraph_freq=5-200_len=6-128.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "bert-large-cased_wiki40b-train-all-paragraph_freq=5-200_len=6-128.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, unbiased, freq=5~200, length=6~128."
    },
    "wiki40b-all-narrow-vocab": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "bert-large-cased_wiki40b-train-all-paragraph_freq=100-300_len=6-128.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "bert-large-cased_wiki40b-train-all-paragraph_freq=100-300_len=6-128.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, freq=100~300, length=6~128."
    },
    "wiki40b-all-ext": {
        "path":os.path.join(DIR_EMBEDDINGS, "bert-large-cased_wiki40b-train-all-paragraph_freq=100-200_len=6-128_random=False.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, freq=100~200, length=6~128, disable random sampling."
    },
}
