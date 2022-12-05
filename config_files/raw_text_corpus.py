#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
from .utils import pick_first_available_path
from dataset.filter import EmptyFilter, SequenceLengthFilter
from dataset.deprecated import FrequencyBasedMonosemousEntitySampler

DIR_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/raw_text_corpus/"
DIR_EMBEDDINGS = os.path.join(DIR_CORPUS, "./bert_embeddings/")
DIR_LOCAL = "/tmp/sakae/"

# We use NDJSONDataset class
# we also recommend EmptyFilter, SequenceLengthFilter, and FrequencyBasedMonosemousEntitySampler.
# EmptyFilter: remove sentence with no annotation.
# SequenceLengthFilter: limit min/max sequence length.
# FrequencyBasedMonosemousEntitySampler: limit number of occurence
cfg_corpus = {
    "wikitext103": {
        "path": os.path.join(DIR_CORPUS, "wikitext103_train_pos-all.jsonl"),
        "binary": False,
        "description": "Wikitext103 trainset > PoS tagging, lemmatization + MWE extraction."
    },
    "wiki40b_all_paragraph": {
        "path": os.path.join(DIR_CORPUS, "wiki40b_train_all_paragraph_lemma-pos-mwe.json.bin"),
        "binary": True,
        "description": "Wiki40b whole dataset > PoS tagging, lemmatization + MWE extraction.",
    },
    "wiki40b_first_paragraph": {
        "path": os.path.join(DIR_CORPUS, "wiki40b_train_first_paragraph_lemma-pos-mwe.json"),
        "binary": False,
        "description": "Wiki40b subset (first paragraph only) > PoS tagging, lemmatization + MWE extraction.",
    },
}

cfg_embeddings = {
    "sample": {
        "path":os.path.join(DIR_EMBEDDINGS, "wiki40b_sample_bert-large-cased.hdf5"),
        "max_sequence_length": None,
        "is_context_embeddings_in_entity_only": True,
        "description": "BERT-large-cased. sample dataset for development purpose."
    },
    "wikitext103": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "wikitext103_train_bert-large-cased.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "wikitext103_train_bert-large-cased.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "is_context_embeddings_in_entity_only": True,
        "description": "BERT-large-cased. Wikitext-103 trainset, context-sense similarity added, length=6~128."
    },
    "wiki40b_first_paragraph": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "wiki40b_train_first_paragraph_bert-large-cased.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "wiki40b_train_first_paragraph_bert-large-cased.hdf5"),
        ),
        "padding": False,
        "max_sequence_length": None,
        "is_context_embeddings_in_entity_only": True,
        "description": "BERT-large-cased. wiki40b trainset(first paragraph), context-sense similarity added, length=6~128."
    }
}
