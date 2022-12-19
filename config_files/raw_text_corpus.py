#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional

import sys, io, os
from .utils import pick_first_available_path
from dataset.filter import EmptyFilter
from dataset.transform import SenseFrequencyBasedEntitySampler

DIR_CORPUS = "/home/sakae/Windows/dataset/word_sense_disambiguation/raw_text_corpus/"
DIR_EMBEDDINGS = os.path.join(DIR_CORPUS, "./bert_embeddings/")
DIR_LOCAL = "/tmp/sakae/"

# We use NDJSONDataset class
# we also recommend EmptyFilter and SenseFrequencyBasedEntitySampler.
# EmptyFilter: remove sentences with no annotation.
# SenseFrequencyBasedEntitySampler: downsample most similar senses.
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
        "path_sense_freq": os.path.join(DIR_EMBEDDINGS, "wiki40b_sample_bert-large-cased.hdf5.sense_freq.json"),
        "max_sequence_length": None,
        "is_context_embeddings_in_entity_only": True,
        "description": "BERT-large-cased. sample dataset for development purpose."
    },
    "wikitext103": {
        "path":pick_first_available_path(
            os.path.join(DIR_LOCAL, "wikitext103_train_bert-large-cased.hdf5"),
            os.path.join(DIR_EMBEDDINGS, "wikitext103_train_bert-large-cased.hdf5"),
        ),
        "path_sense_freq": None,
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
        "path_sense_freq": None,
        "padding": False,
        "max_sequence_length": None,
        "is_context_embeddings_in_entity_only": True,
        "description": "BERT-large-cased. wiki40b trainset(first paragraph), context-sense similarity added, length=6~128."
    }
}

def setup_neighbor_sense_downsampler(path_sense_freq: str,
                                     min_freq: Optional[int] = 1, max_freq: Optional[int] = 10,
                                     enable_random_sampling: bool = True,
                                     entity_field_name: str = "record.entities",
                                     lemma_key_field_name: str = "most_similar_sense_lemma_key",
                                     is_multiple_senses: bool = False
                                    ):
    """
    BERTEmbeddingsDatasetに対して語義をダウンサンプリングするfilter,transformerを返す．
    usage:
    dict_filter_transformer = setup_neighbor_sense_downsampler(...)
    dataset = BERTEmbeddingsDataset(..., **dict_filter_transformer)

    Args:
        path_sense_freq: 語義頻度データ（key: lemma_key, value: freq）のパス
        min_freq: 最低頻度．指定値未満の語義はすべて無視．DEFAULT: 1 (=すべての語義)
        max_freq: 最大頻度．指定値より高頻度の語義はダウンサンプリングされる．DEFAULT: 10 (=高々10回)
        enable_random_sampling: ダウンサンプリングをランダム化する(=True)または出現順にする(=False)．乱択はエポックごとに行う．

    Returns: SenseFrequencyBasedEntitySamplerとEmptyFilterのDict

    """
    entity_sampler = SenseFrequencyBasedEntitySampler(
        path_sense_freq=path_sense_freq,
        min_freq=min_freq, max_freq=max_freq, enable_random_sampling=enable_random_sampling, random_seed=42,
        entity_field_name=entity_field_name, lemma_key_field_name=lemma_key_field_name, is_multiple_senses=is_multiple_senses
    )

    dict_dataset_args = {
        "filter_function": EmptyFilter(check_field_names=[entity_field_name]),
        "transform_functions": {entity_field_name: entity_sampler}
    }
    return dict_dataset_args
