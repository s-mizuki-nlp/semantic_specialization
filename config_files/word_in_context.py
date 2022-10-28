#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os

DIR_DATASET_WiC = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/"
DIR_DATASET_WiC_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/bert_embeddings/"

# MCL-WiC Dataset [Martelli+, 2021]
# We use English dataset only.
DIR_DATASET_MCL_WiC = "/home/sakae/Windows/dataset/word_sense_disambiguation/mcl_wic/"
DIR_DATASET_MCL_WiC_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/mcl_wic/bert_embeddings/"


cfg_corpus = {
    "WiC-train": {
        "path_corpus": os.path.join(DIR_DATASET_WiC, "./train/train.data.txt"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET_WiC, "./train/train.gold.txt"),
    },
    "WiC-dev": {
        "path_corpus": os.path.join(DIR_DATASET_WiC, "./dev/dev.data.txt"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET_WiC, "./dev/dev.gold.txt"),
    },
    "WiC-test": {
        "path_corpus": os.path.join(DIR_DATASET_WiC, "./test/test.data.txt"),
        "path_ground_truth_labels": None,
    },
    "MCL-WiC-train": {
        "path_corpus": os.path.join(DIR_DATASET_MCL_WiC, "./training/training.en-en.data"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET_MCL_WiC, "./training/training.en-en.gold"),
        "fix_wrong_annotation": True,
    },
    "MCL-WiC-dev": {
        "path_corpus": os.path.join(DIR_DATASET_MCL_WiC, "./dev/multilingual/dev.en-en.data"),
        "path_ground_truth_labels": os.path.join(DIR_DATASET_MCL_WiC, "./dev/multilingual/dev.en-en.gold")
    },
    "MCL-WiC-test": {
        "path_corpus": os.path.join(DIR_DATASET_MCL_WiC, "./test/multilingual/test.en-en.data"),
        "path_ground_truth_labels": None
    }
}

cfg_tasks = {
    "WiC-train": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_WiC_EMBEDDINGS, "bert-large-cased_WiC-train.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task trainset encoded by BERT-large-cased."
    },
    "WiC-dev": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_WiC_EMBEDDINGS, "bert-large-cased_WiC-dev.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task development set encoded by BERT-large-cased."
    },
    "WiC-test": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_WiC_EMBEDDINGS, "bert-large-cased_WiC-test.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Word-in-Context task test set encoded by BERT-large-cased."
    },
    "MCL-WiC-train": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_MCL_WiC_EMBEDDINGS, "bert-large-cased_MCL-WiC-train.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Multilingual Word-in-Context task trainset for En-En encoded by BERT-large-cased."
    },
    "MCL-WiC-dev": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_MCL_WiC_EMBEDDINGS, "bert-large-cased_MCL-WiC-dev.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Multilingual Word-in-Context task dev. set for En-En encoded by BERT-large-cased."
    },
    "MCL-WiC-test": {
        "path_or_bert_embeddings_dataset": os.path.join(DIR_DATASET_MCL_WiC_EMBEDDINGS, "bert-large-cased_MCL-WiC-test.hdf5"),
        "kwargs_bert_embeddings_dataset": {"padding":False, "max_sequence_length": None},
        "copy_field_names_from_record_to_entity": ["sentence_pair_id", "tokenized_sentence", "words", "ground_truth_label"],
        "description": "Multilingual Word-in-Context task test set for En-En encoded by BERT-large-cased. Ground-truth label is unavailable."
    },
}

