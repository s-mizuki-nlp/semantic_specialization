#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os

DIR_DATASET = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/"
DIR_DATASET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/word_in_context/bert_embeddings/"

cfg_cropus = {
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