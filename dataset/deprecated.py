#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import json
import os, sys, io
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset
from dataset.utils import lemma_pos_to_tuple

class FrequencyBasedMonosemousEntitySampler(object):

    def __init__(self,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,
                 path_monosemous_words_freq: Optional[str] = None,
                 dataset_monosemous_entity_annotated_corpus: Optional[Dataset] = None,
                 lemma_lowercase: bool = True,
                 enable_random_sampling: bool = True,
                 entity_field_name: str = "entities",
                 random_seed: int = 42
                 ):
        """
        単義語をsamplingする．指定頻度未満の単義語は削除，指定頻度以上の単義語はdown-samplingする．

        @param min_freq: 最小頻度．指定値未満の単義語は削除．
        @param max_freq: 最大頻度．指定値を上回る単義語は，先頭max_freq個のみ採択(enable_random_sampling=False) または 確率的に採択(enable_random_sampling=True)
        @param path_monosemous_words_freq: 単義語の頻度データ．フォーマットはNDJSON, `{'lemma': 'toddler', 'pos': 'n', 'freq': 2681}`
        @param enable_random_sampling: True: 最大頻度を上回る単義語について確率的に採択． False: 先頭max_freq個のみ採択．

        @rtype: object
        """

        if isinstance(min_freq, int):
            if path_monosemous_words_freq is not None:
                assert os.path.exists(path_monosemous_words_freq), f"specified file does not exist: {path_monosemous_words_freq}"
                self._lemma_pos_freq = self._load_lemma_pos_freq(path=path_monosemous_words_freq)
            elif dataset_monosemous_entity_annotated_corpus is not None:
                print("counting lemma x pos frequency from dataset.")
                self._lemma_pos_freq = self._count_lemma_pos_freq(dataset=dataset_monosemous_entity_annotated_corpus,
                                                                  lemma_lowercase=lemma_lowercase, entity_field_name=entity_field_name)
            else:
                raise AssertionError(f"you must specify either `path_monosemous_words_freq` or `dataset_monosemous_entity_annotated_corpus`.")
        else:
            # it always returns zero.
            self._lemma_pos_freq = defaultdict(int)

        self._min_freq = 0 if min_freq is None else min_freq
        self._max_freq = float("inf") if max_freq is None else max_freq
        self._lemma_lowercase = lemma_lowercase
        self._enable_random_sampling = enable_random_sampling
        self._entity_field_name = entity_field_name

        if enable_random_sampling:
            np.random.seed(random_seed)
            self._cursor = -1
            self._random_values = np.random.uniform(size=2 ** 24)

        self.reset()

    def _load_lemma_pos_freq(self, path: str):
        dict_freq = {}
        ifs = io.open(path, mode="r")
        for record in ifs:
            d = json.loads(record.strip())
            key = (d["lemma"], d["pos"])
            dict_freq[key] = d["freq"]
        ifs.close()

        return dict_freq

    @classmethod
    def _count_lemma_pos_freq(cls, dataset: Dataset, lemma_lowercase: bool, entity_field_name: str):
        cnt = Counter()
        for record in dataset:
            lst_lemma_pos = [lemma_pos_to_tuple(lemma_lowercase=lemma_lowercase, **entity) for entity in record[entity_field_name]]
            cnt.update(lst_lemma_pos)

        return cnt

    def random_uniform(self) -> float:
        if self._enable_random_sampling:
            self._cursor += 1
            self._cursor %= self._random_values.size
            return self._random_values[self._cursor]
        else:
            return 0.0

    def decide_sample_or_not(self, lemma_pos):
        freq_total = self._lemma_pos_freq[lemma_pos]
        freq_sampled = self._lemma_pos_freq_sampled[lemma_pos]
        freq_missed = self._lemma_pos_freq_missed[lemma_pos]
        freq_remain = freq_total - freq_missed

        if  (freq_total < self._min_freq) or (freq_sampled >= self._max_freq):
            is_sampled = False
        elif freq_remain <= self._max_freq:
            is_sampled = True
        else:
            # sampling based on total frequency
            is_sampled = (self._max_freq / freq_total) > self.random_uniform()

        if is_sampled:
            self._lemma_pos_freq_sampled[lemma_pos] += 1
        else:
            self._lemma_pos_freq_missed[lemma_pos] += 1

        return is_sampled

    def __call__(self, lst_entities: List[Dict[str, str]]):
        """
        @param lst_entities: list of dict like `[{'lemma': 'Dubonnet', 'pos': 'n', 'occurence': 0, 'span': [1, 2]}]`
        """
        lst_ret = []
        for entity in lst_entities:
            lemma_pos = lemma_pos_to_tuple(lemma_lowercase=self._lemma_lowercase, **entity)
            if not self.decide_sample_or_not(lemma_pos):
                continue
            lst_ret.append(entity)

        return lst_ret

    def __getitem__(self, lemma_pos: Tuple[str, str]):
        key = lemma_pos_to_tuple(lemma_pos[0], lemma_pos[1], self._lemma_lowercase)
        return self._lemma_pos_freq[key]

    def _export_num_of_valid_lemma_pos(self):
        dict_cnt = defaultdict(int)
        for (lemma, pos), freq in self._lemma_pos_freq.items():
            if freq >= self._min_freq:
                dict_cnt[pos] += 1
        return dict_cnt

    def _export_freq_of_valid_lemma_pos(self):
        dict_cnt = defaultdict(int)
        for (lemma, pos), freq in self._lemma_pos_freq.items():
            if freq >= self._min_freq:
                dict_cnt[pos] += min(self._max_freq, freq)
        return dict_cnt

    def reset(self):
        self._lemma_pos_freq_sampled = defaultdict(int)
        self._lemma_pos_freq_missed = defaultdict(int)

    def verbose(self):
        ret = {
            "min_freq": self._min_freq,
            "max_freq": self._max_freq,
            "lemma_lowercase": self._lemma_lowercase,
            "n_total_lemma_and_pos_vocab": len(self._lemma_pos_freq),
            "n_valid_lemma_and_pos_vocab": self._export_num_of_valid_lemma_pos(),
            "n_valid_lemma_and_pos_freq": self._export_freq_of_valid_lemma_pos()
        }
        return ret
