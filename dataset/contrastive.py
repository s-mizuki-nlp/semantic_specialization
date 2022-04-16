#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Union, List, Optional, Callable, Iterable, Dict, Any
import os, sys, io
import copy, pickle, random
from collections import defaultdict
from functools import lru_cache
import warnings
from tqdm import tqdm

import numpy as np
from nltk import word_tokenize
from nltk.corpus import wordnet as wn

from torch.utils.data import Dataset, IterableDataset
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss
from .gloss_sref import SREFLemmaEmbeddingsDataset, WordNetGlossDataset
from .sense_expansion import extract_lemma_keys_and_weights_from_semantically_related_synsets

class ContrastiveLearningDataset(IterableDataset):

    def __init__(self, corpus_dataset: Union[SREFLemmaEmbeddingsDataset, WordNetGlossDataset],
                 iterate_over_lemma_or_lemma_key: str = "lemma_key",
                 semantic_relation_for_positives: str = "all-relations",
                 use_taxonomy_distance_for_sampling_positives: bool = True,
                 shuffle: bool = True,
                 num_hard_negatives: int = 0):
        """

        Args:
            corpus_dataset:
            semantic_relation_for_positives: "all-relations", "all-relations-but-hyponymy", "all-relations-but-synonymy", "hyponymy"
            num_hard_negatives: number of hard negative examples these are homographs of the target lemma. 0: None, -1: all homographs, N: as many as N homographs.
        """
        self._corpus_dataset = corpus_dataset
        self._num_hard_negatives = num_hard_negatives
        self._semantic_relation_for_positives = semantic_relation_for_positives
        self._use_taxonomy_distance_for_sampling_positives = use_taxonomy_distance_for_sampling_positives
        self._iterate_over_lemma_or_lemma_key = iterate_over_lemma_or_lemma_key
        self._shuffle = shuffle

    def __len__(self):
        if self._iterate_over_lemma_or_lemma_key == "lemma_key":
            return len(self._corpus_dataset.get_lemma_keys())
        elif self._iterate_over_lemma_or_lemma_key == "lemma":
            return len(self._corpus_dataset.get_lemma_and_pos())

    def iter_by_lemma_keys(self, shuffle: bool = True):
        """
        iterate over all lemma keys once.
        """
        lst_lemma_keys = self._corpus_dataset.get_lemma_keys()
        if shuffle:
            random.shuffle(lst_lemma_keys)

        for lemma_key in lst_lemma_keys:
            record = self.get_contrastive_example(query_lemma_key=lemma_key)
            if record is not None:
                yield record

    def iter_by_lemma_and_pos(self, shuffle: bool = True):
        lst_tup_lemma_and_pos = self._corpus_dataset.get_lemma_and_pos()
        if shuffle:
            random.shuffle(lst_tup_lemma_and_pos)

        for lemma, pos in lst_tup_lemma_and_pos:
            record = self.get_contrastive_example(lemma, pos)
            if record is not None:
                yield record

    def __iter__(self):
        if self._iterate_over_lemma_or_lemma_key == "lemma_key":
            return self.iter_by_lemma_keys(shuffle=self._shuffle)
        elif self._iterate_over_lemma_or_lemma_key == "lemma":
            return self.iter_by_lemma_and_pos(shuffle=self._shuffle)

    def get_contrastive_example(self, lemma: Optional[str] = None, pos: Optional[str] = None, query_lemma_key: Optional[str] = None):

        # query
        if query_lemma_key is None:
            lst_lemma_keys = self._corpus_dataset.get_lemma_keys_by_lemma_and_pos(lemma, pos)
            query_lemma_key = random.choice(lst_lemma_keys)
        else:
            if lemma is None:
                lemma = utils_wordnet_gloss.lemma_key_to_lemma_name(query_lemma_key)
            if pos is None:
                pos = utils_wordnet_gloss.lemma_key_to_pos(query_lemma_key, tagtype="short") # it returns one of ["n","v","s","r"]
            lst_lemma_keys = self._corpus_dataset.get_lemma_keys_by_lemma_and_pos(lemma, pos)

        synset_id = utils_wordnet_gloss.lemma_key_to_synset_id(query_lemma_key)

        # positives = semantically related lemma keys of the query.
        lst_positive_lemma_keys, lst_positive_weights = extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id=synset_id,
                                                                                                                         semantic_relation=self._semantic_relation_for_positives,
                                                                                                                         distinct=True, fix_synonym_distance=True)
        if query_lemma_key in lst_positive_lemma_keys:
            idx = lst_positive_lemma_keys.index(query_lemma_key)
            del lst_positive_lemma_keys[idx]
            del lst_positive_weights[idx]

        # sanity check
        if len(lst_positive_lemma_keys) == 0:
            warnings.warn(f"no semantically related lemma: {lemma}|{pos}::{query_lemma_key}")
            return None

        if self._use_taxonomy_distance_for_sampling_positives:
            v_probs = np.array(lst_positive_weights) / sum(lst_positive_weights)
            positive_lemma_key = np.random.choice(lst_positive_lemma_keys, p=v_probs)
        else:
            positive_lemma_key = random.choice(lst_positive_lemma_keys)

        # hard negatives = homographs of the query.
        if self._num_hard_negatives == 0:
            lst_hard_negative_lemma_keys = []
        else:
            lst_hard_negative_lemma_keys = copy.deepcopy(lst_lemma_keys)
            # remove query and positive (if exists)
            lst_hard_negative_lemma_keys.remove(query_lemma_key)
            if positive_lemma_key in lst_hard_negative_lemma_keys:
                lst_hard_negative_lemma_keys.remove(positive_lemma_key)

            if self._num_hard_negatives > 0:
                # choose as many as N lemma keys randomly.
                random.shuffle(lst_hard_negative_lemma_keys)
                lst_hard_negative_lemma_keys = lst_hard_negative_lemma_keys[:self._num_hard_negatives]

        # prepare contrastive example object
        lst_hard_negatives = []
        for lemma_key in lst_hard_negative_lemma_keys:
            lst_hard_negatives.extend(self._corpus_dataset.get_records_by_lemma_key(lemma_key))

        dict_result = {
            "query": self._corpus_dataset.get_records_by_lemma_key(query_lemma_key)[0],
            "positive": self._corpus_dataset.get_records_by_lemma_key(positive_lemma_key)[0],
            "hard_negatives": lst_hard_negatives,
            "num_hard_negatives": len(lst_hard_negatives)
        }

        return dict_result

    @property
    def verbose(self):
        lst_attr_names = "num_hard_negatives,semantic_relation_for_positives,use_taxonomy_distance_for_sampling_positives,iterate_over_lemma_or_lemma_key,shuffle".split(",")
        ret = {attr_name:getattr(self, "_" + attr_name) for attr_name in lst_attr_names}
        ret["__len__"] = self.__len__()
        ret["corpus_dataset"] = self._corpus_dataset.verbose
        return ret