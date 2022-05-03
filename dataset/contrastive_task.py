#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Union, Optional
import copy, random
import warnings

import numpy as np

from torch.utils.data import Dataset
from dataset_preprocessor import utils_wordnet_gloss
from .gloss_embeddings import SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset
from .gloss import WordNetGlossDataset
from .sense_expansion import extract_lemma_keys_and_weights_from_semantically_related_synsets

class ContrastiveLearningDataset(Dataset):

    def __init__(self, gloss_dataset: Union[SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset, WordNetGlossDataset],
                 iterate_over_lemma_or_lemma_key: str = "lemma_key",
                 semantic_relation_for_positives: str = "all-relations",
                 use_taxonomy_distance_for_sampling_positives: bool = True,
                 num_hard_negatives: int = 0):
        """

        Args:
            gloss_dataset:
            semantic_relation_for_positives: "all-relations", "all-relations-but-hyponymy", "all-relations-but-synonymy", "hyponymy"
            num_hard_negatives: number of hard negative examples these are homographs of the target lemma. 0: None, -1: all homographs, N: as many as N homographs.
        """
        self._gloss_dataset = gloss_dataset
        self._num_hard_negatives = num_hard_negatives
        self._semantic_relation_for_positives = semantic_relation_for_positives
        self._use_taxonomy_distance_for_sampling_positives = use_taxonomy_distance_for_sampling_positives
        self._iterate_over_lemma_or_lemma_key = iterate_over_lemma_or_lemma_key

    def clean_up_invalid_items(self, lst_items):
        lst_valid_items = []
        for lemma_key_or_lemma_and_pos in lst_items:
            if self._iterate_over_lemma_or_lemma_key == "lemma_key":
                record = self.get_contrastive_example(query_lemma_key=lemma_key_or_lemma_and_pos)
            elif self._iterate_over_lemma_or_lemma_key == "lemma":
                record = self.get_contrastive_example(lemma=lemma_key_or_lemma_and_pos[0], pos=lemma_key_or_lemma_and_pos[1])
            if record is not None:
                lst_valid_items.append(lemma_key_or_lemma_and_pos)
        return lst_valid_items

    @property
    def items(self):
        if hasattr(self, "_items"):
            return self._items
        else:
            if self._iterate_over_lemma_or_lemma_key == "lemma_key":
                lst_items = self._gloss_dataset.get_lemma_keys()
            elif self._iterate_over_lemma_or_lemma_key == "lemma":
                lst_items = self._gloss_dataset.get_lemma_and_pos()

            # clean un invalid (=do not have semantically related) items
            self._items = self.clean_up_invalid_items(lst_items)

        return self._items

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        if self._iterate_over_lemma_or_lemma_key == "lemma_key":
            lemma_key = self.items[idx]
            record = self.get_contrastive_example(query_lemma_key=lemma_key, suppress_warning=True)
        else:
            lemma, pos = self.items[idx]
            record = self.get_contrastive_example(lemma=lemma, pos=pos, suppress_warning=True)
        return record

    def get_contrastive_example(self, lemma: Optional[str] = None, pos: Optional[str] = None, query_lemma_key: Optional[str] = None, suppress_warning: bool = False):
        # query
        if query_lemma_key is None:
            lst_lemma_keys = self._gloss_dataset.get_lemma_keys_by_lemma_and_pos(lemma, pos)
            query_lemma_key = random.choice(lst_lemma_keys)
        else:
            if lemma is None:
                lemma = utils_wordnet_gloss.lemma_key_to_lemma_name(query_lemma_key)
            if pos is None:
                pos = utils_wordnet_gloss.lemma_key_to_pos(query_lemma_key, tagtype="short") # it returns one of ["n","v","s","r"]
            lst_lemma_keys = self._gloss_dataset.get_lemma_keys_by_lemma_and_pos(lemma, pos)

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
            if not suppress_warning:
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
            lst_hard_negatives.extend(self._gloss_dataset.get_records_by_lemma_key(lemma_key))

        dict_result = {
            "query": self._gloss_dataset.get_records_by_lemma_key(query_lemma_key)[0],
            "positive": self._gloss_dataset.get_records_by_lemma_key(positive_lemma_key)[0],
            "hard_negatives": lst_hard_negatives,
            "num_hard_negatives": len(lst_hard_negatives)
        }

        return dict_result

    @property
    def verbose(self):
        lst_attr_names = "num_hard_negatives,semantic_relation_for_positives,use_taxonomy_distance_for_sampling_positives,iterate_over_lemma_or_lemma_key,shuffle".split(",")
        ret = {attr_name:getattr(self, "_" + attr_name) for attr_name in lst_attr_names}
        ret["__len__"] = self.__len__()
        ret["corpus_dataset"] = self._gloss_dataset.verbose
        return ret

    @property
    def gloss_dataset(self):
        return self._gloss_dataset