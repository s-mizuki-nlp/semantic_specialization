#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Callable, List, Optional
import sys, io, os, json
from collections import defaultdict, Counter
import itertools
import pydash

import numpy as np
from torch.utils.data import Dataset

from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss



class FieldTypeConverter(object):

    def __init__(self, dict_field_type_converter: Dict[str, Callable]):

        self._dict_field_type_converter = dict_field_type_converter

    def __call__(self, sample):

        for field_name, converter in self._dict_field_type_converter.items():
            if field_name in sample:
                sample[field_name] = converter(sample[field_name])

        return sample


class ToWordNetPoSTagConverter(object):

    def __init__(self, pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos",
                 convert_adjective_to_adjective_satellite: bool = True,
                 keep_original_pos: bool = False):

        self._pos_field_name = pos_field_name
        self._mwe_pos_field_name = mwe_pos_field_name
        self._convert_adjective_to_adjective_satellite = convert_adjective_to_adjective_satellite
        self._keep_original_pos = keep_original_pos
        self._pos_orig_field_name = pos_field_name + "_orig"
        self._mwe_pos_orig_field_name = mwe_pos_field_name + "_orig"

    def _adjective_to_adjective_satellite(self, pos):
        return "s" if pos == "a" else pos

    def __call__(self, lst_tokens):
        # Stanford token PoS tags to WordNet PoS tags.

        for token in lst_tokens:
            if self._keep_original_pos:
                token[self._pos_orig_field_name] = token[self._pos_field_name]
            pos = utils_wordnet.ptb_tagset_to_wordnet_tagset(token[self._pos_field_name])
            if self._convert_adjective_to_adjective_satellite:
                pos = self._adjective_to_adjective_satellite(pos)
            token[self._pos_field_name] = pos

            if self._mwe_pos_field_name in token:
                if self._keep_original_pos:
                    token[self._mwe_pos_orig_field_name] = token[self._mwe_pos_field_name]
                pos = utils_wordnet.jmwe_tagset_to_wordnet_tagset(token[self._mwe_pos_field_name])
                if self._convert_adjective_to_adjective_satellite:
                    pos = self._adjective_to_adjective_satellite(pos)
                token[self._mwe_pos_field_name] = pos

        return lst_tokens


class ToWordNetPoSTagAndLemmaConverter(ToWordNetPoSTagConverter):

    def __init__(self,
                 pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos",
                 lemma_field_name: str = "lemma",
                 mwe_lemma_field_name: str = "mwe_lemma",
                 lemma_and_pos_field_name: str = "lemma_pos",
                 mwe_lemma_and_pos_field_name: str = "mwe_lemma_pos",
                 convert_adjective_to_adjective_satellite: bool = True,
                 keep_original_pos: bool = False,
                 lowercase: bool = True):

        super().__init__(pos_field_name, mwe_pos_field_name, convert_adjective_to_adjective_satellite, keep_original_pos)
        self._lemma_field_name = lemma_field_name
        self._mwe_lemma_field_name = mwe_lemma_field_name
        self._lemma_and_pos_field_name = lemma_and_pos_field_name
        self._mwe_lemma_and_pos_field_name = mwe_lemma_and_pos_field_name
        self._lowercase = lowercase
        self._keep_original_pos = keep_original_pos

    def __call__(self, lst_tokens):
        lst_tokens = super().__call__(lst_tokens)
        if self._lowercase:
            return self._create_tuple_lowercase(lst_tokens)
        else:
            return self._create_tuple(lst_tokens)

    def _create_tuple_lowercase(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name].lower(), token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name].lower(), token[self._mwe_pos_field_name])

        return lst_tokens

    def _create_tuple(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name], token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name], token[self._mwe_pos_field_name])

        return lst_tokens


class WordNetPoSTagAndLemmaToEntityConverter(object):

    def __init__(self, sense_inventory: "WordNetGlossCorpus"):
        self._sense_inventory = sense_inventory

    def __call__(self, lst_dict_tokens: List[Dict[str, str]]):
        """
        @param lst_dict_tokens: list of token dict like `[{'lemma': 'video', 'pos': 'n', "mwe_lemma":"video_game", "mwe_pos":"n"}]`
        """

        lst_entities = []
        idx = 0

        while idx < len(lst_dict_tokens):
            dict_token = lst_dict_tokens[idx]

            # Multi-word expression
            if "mwe_pos" in dict_token:
                lst_candidate_lemma_keys = self._sense_inventory.get_lemma_keys_by_lemma_and_pos(lemma=dict_token["mwe_lemma"], pos=dict_token["mwe_pos"])
                if len(lst_candidate_lemma_keys) > 0:
                    dict_entity = {
                        "candidate_lemma_keys": lst_candidate_lemma_keys,
                        "lemma": dict_token["mwe_lemma"],
                        "pos": dict_token["mwe_pos"],
                        "pos_orig": dict_token["mwe_pos_orig"],
                        "span": dict_token["mwe_span"],
                    }
                    lst_entities.append(dict_entity)
                    # jump to the end of MWE
                    idx = dict_token["mwe_span"][1]
                    continue

            # Not MWE
            lst_candidate_lemma_keys = self._sense_inventory.get_lemma_keys_by_lemma_and_pos(lemma=dict_token["lemma"], pos=dict_token["pos"])
            if len(lst_candidate_lemma_keys) > 0:
                dict_entity = {
                    "candidate_lemma_keys": lst_candidate_lemma_keys,
                    "lemma": dict_token["lemma"],
                    "pos": dict_token["pos"],
                    "pos_orig": dict_token["pos_orig"],
                    "span": [idx, idx + 1]
                }
                lst_entities.append(dict_entity)

            idx += 1

        return lst_entities


class SenseFrequencyBasedEntitySampler(object):

    def __init__(self,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,
                 path_sense_freq: Optional[str] = None,
                 dataset_sense_annotated_corpus: Optional[Dataset] = None,
                 enable_random_sampling: bool = True,
                 entity_field_name: str = "record.entities",
                 lemma_key_field_name: str = "most_similar_sense_lemma_key",
                 is_multiple_senses: bool = False,
                 random_seed: int = 42
                 ):
        """
        語義をsamplingする．指定頻度未満の語義は削除，指定頻度以上の語義はdown-samplingする．

        @param min_freq: 最小頻度．
        @param max_freq: 最大頻度．指定値を上回る語義は，先頭max_freq個のみ採択(enable_random_sampling=False) または 確率的に採択(enable_random_sampling=True)
        @param path_sense_freq: 語義の頻度データ．keyはlemma key, valueは頻度．フォーマットはJSON, `{'active%3:00:02::':15, 'balloonist%1:18:00::':123, ...}`
        @param enable_random_sampling: True: 最大頻度を上回る語義について確率的に採択． False: 先頭max_freq個のみ採択．

        @rtype: object
        """

        if isinstance(min_freq, int):
            if path_sense_freq is not None:
                assert os.path.exists(path_sense_freq), f"specified file does not exist: {path_sense_freq}"
                self._lemma_key_freq = self._load_lemma_key_freq(path=path_sense_freq)
            elif dataset_sense_annotated_corpus is not None:
                print("counting sense key frequency from dataset.")
                self._lemma_key_freq = self._count_lemma_key_freq(dataset=dataset_sense_annotated_corpus,
                                                                  entity_field_name=entity_field_name,
                                                                  lemma_key_field_name=lemma_key_field_name,
                                                                  is_multiple_senses=is_multiple_senses)
            else:
                raise AssertionError(f"you must specify either `path_sense_freq` or `dataset_sense_annotated_corpus`.")
        else:
            # it always returns zero.
            self._lemma_key_freq = defaultdict(int)

        self._min_freq = 0 if min_freq is None else min_freq
        self._max_freq = float("inf") if max_freq is None else max_freq
        self._enable_random_sampling = enable_random_sampling
        self._entity_field_name = entity_field_name
        self._lemma_key_field_name = lemma_key_field_name

        if enable_random_sampling:
            np.random.seed(random_seed)
            self._cursor = -1
            self._random_values = np.random.uniform(size=2 ** 24)

        self.reset()

    def _load_lemma_key_freq(self, path: str) -> Dict[str, int]:
        with io.open(path, mode="r") as ifs:
            dict_freq = json.load(ifs)
        return dict_freq

    def save_lemma_key_freq(self, path: str) -> bool:
        if os.path.exists(path):
            raise IOError(f"specified file already exist: {path}")

        with io.open(path, mode="w") as ofs:
            json.dump(self._lemma_key_freq, ofs)

        return True

    @classmethod
    def _count_lemma_key_freq(cls, dataset: Dataset, entity_field_name: str, lemma_key_field_name: str,
                              is_multiple_senses: bool = False):
        cnt = Counter()
        is_bert_embeddings_dataset = dataset.__class__.__name__ == "BERTEmbeddingsDataset"

        if is_bert_embeddings_dataset:
            dataset.return_record_only = True

        for record in dataset:
            lst_lemma_keys = [entity[lemma_key_field_name] for entity in pydash.get(record, entity_field_name)]
            if is_multiple_senses:
                lst_lemma_keys = list(itertools.chain(*lst_lemma_keys))
            cnt.update(lst_lemma_keys)

        if is_bert_embeddings_dataset:
            dataset.return_record_only = False

        return cnt

    def random_uniform(self) -> float:
        if self._enable_random_sampling:
            self._cursor += 1
            self._cursor %= self._random_values.size
            return self._random_values[self._cursor]
        else:
            return 0.0

    def decide_sample_or_not(self, lemma_key):
        freq_total = self._lemma_key_freq[lemma_key]
        freq_sampled = self._freq_sampled[lemma_key]
        freq_missed = self._freq_missed[lemma_key]
        freq_remain = freq_total - freq_missed

        if  (freq_total < self._min_freq) or (freq_sampled >= self._max_freq):
            is_sampled = False
        elif freq_remain <= self._max_freq:
            is_sampled = True
        else:
            # sampling based on total frequency
            is_sampled = (self._max_freq / freq_total) > self.random_uniform()

        if is_sampled:
            self._freq_sampled[lemma_key] += 1
        else:
            self._freq_missed[lemma_key] += 1

        return is_sampled

    def __call__(self, lst_entities: List[Dict[str, str]]):
        """
        @param lst_entities: list of dict like `[{'lemma': 'Dubonnet', 'pos': 'n', 'lemma_key': 'review%2:31:00::', 'span': [1, 2]}]`
        """
        lst_ret = []
        for entity in lst_entities:
            lemma_key = entity[self._lemma_key_field_name]
            if not self.decide_sample_or_not(lemma_key=lemma_key):
                continue
            lst_ret.append(entity)

        return lst_ret

    def __getitem__(self, lemma_key: str):
        return self._lemma_key_freq[lemma_key]

    def _export_num_of_valid_lemma_key_by_pos(self):
        dict_cnt = defaultdict(int)
        for lemma_key, freq in self._lemma_key_freq.items():
            pos = utils_wordnet_gloss.lemma_key_to_pos(lemma_sense_key=lemma_key, tagtype="short")
            if freq >= self._min_freq:
                dict_cnt[pos] += 1
        return dict_cnt

    def _export_freq_of_valid_lemma_key_by_pos(self):
        dict_cnt = defaultdict(int)
        for lemma_key, freq in self._lemma_key_freq.items():
            pos = utils_wordnet_gloss.lemma_key_to_pos(lemma_sense_key=lemma_key, tagtype="short")
            if freq >= self._min_freq:
                dict_cnt[pos] += min(self._max_freq, freq)
        return dict_cnt

    def reset(self):
        self._freq_sampled = defaultdict(int)
        self._freq_missed = defaultdict(int)

    def verbose(self):
        ret = {
            "min_freq": self._min_freq,
            "max_freq": self._max_freq,
            "n_total_lemma_and_pos_vocab": len(self._lemma_key_freq),
            "n_valid_lemma_and_pos_vocab": self._export_num_of_valid_lemma_key_by_pos(),
            "n_valid_lemma_and_pos_freq": self._export_freq_of_valid_lemma_key_by_pos()
        }
        return ret
