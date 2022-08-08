#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os

DIR_DATASET = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/"
DIR_DATASET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/bert_embeddings/"

cfg_corpus = {
    "WiC-train": {
        "path_corpus": os.path.join(DIR_DATASET, "./train/train.data.txt"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET, "./train/train.gold.txt"),
    },
    "WiC-dev": {
        "path_corpus": os.path.join(DIR_DATASET, "./dev/dev.data.txt"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET, "./dev/dev.gold.txt"),
    },
    "WiC-test": {
        "path_corpus": os.path.join(DIR_DATASET, "./test/test.data.txt"),
        "path_ground_truth_labels": None,
    }
}

cfg_tasks = {
    "WiC-train": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_EMBEDDINGS, "bert-large-cased_WiC-train.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task trainset encoded by BERT-large-cased."
    },
    "WiC-dev": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_EMBEDDINGS, "bert-large-cased_WiC-dev.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task development set encoded by BERT-large-cased."
    },
    "WiC-test": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_EMBEDDINGS, "bert-large-cased_WiC-test.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task test set encoded by BERT-large-cased."
    }
}

