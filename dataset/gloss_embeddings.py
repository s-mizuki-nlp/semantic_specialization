#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict, Any, Optional
import io, copy
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from torch.utils.data import Dataset
from dataset.contextualized_embeddings import BERTEmbeddingsDataset
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss
from .encoder import calc_entity_subwords_average_vectors, extract_entity_spans_from_record, tensor_to_numpy

class SREFLemmaEmbeddingsDataset(Dataset):

    def __init__(self, path: str,
                 l2_norm: bool,
                 use_first_embeddings_only: bool = True,
                 lemma_surface_form_lowercase: bool = False,
                 target_pos: List[str] = ["n","v","s","r"],
                 description: str = ""):
        self._path_basic_lemma_embeddings = path
        self._target_pos = target_pos
        self._lemma_surface_form_lowercase = lemma_surface_form_lowercase
        self._description = description

        force_ndim_to_2 = False if use_first_embeddings_only else True
        dict_basic_lemma_embeddings = self.load_basic_lemma_embeddings(path=path, l2_norm=l2_norm, return_first_embeddings_only=use_first_embeddings_only,
                                                                            force_ndim_to_2=force_ndim_to_2)
        # reformat list of records. each record contains single lemma sense key and its embeddings.
        self._dataset = self._annotate_records(dict_basic_lemma_embeddings)
        self._index_by_lemma_and_pos = self._reindex_dataset_using_lemma_and_pos(dataset=self._dataset)
        self._index_by_lemma_key = self._reindex_dataset_using_lemma_key(dataset=self._dataset)
        self._lemma_and_pos_to_lemma_keys = self._map_lemma_and_pos_to_lemma_keys(dataset=self._dataset)


    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for idx in range(len(self)):
            record = self.__getitem__(idx)
            yield record

    def __getitem__(self, item):
        record = self._dataset[item]
        return record

    @classmethod
    def load_basic_lemma_embeddings(cls, path: str, l2_norm: bool, return_first_embeddings_only: bool, force_ndim_to_2: bool = False) -> Dict[str, np.ndarray]:
        dict_lemma_key_embeddings = {}

        with io.open(path, mode="rb") as ifs:
            dict_lst_lemma_embeddings = pickle.load(ifs)

        for lemma_key, lst_or_numpy_lemma_key_embeddings in tqdm(dict_lst_lemma_embeddings.items()):
            if return_first_embeddings_only:
                # DUBIOUS: it just accounts for first embedding of each lemma keys.
                if isinstance(lst_or_numpy_lemma_key_embeddings, list):
                    if isinstance(lst_or_numpy_lemma_key_embeddings[0], list):
                        vectors = np.array(lst_or_numpy_lemma_key_embeddings[0])
                    elif isinstance(lst_or_numpy_lemma_key_embeddings[0], np.ndarray):
                        vectors = np.array(lst_or_numpy_lemma_key_embeddings[0])
                    else:
                        vectors = np.array(lst_or_numpy_lemma_key_embeddings)
                elif isinstance(lst_or_numpy_lemma_key_embeddings, np.ndarray):
                    if lst_or_numpy_lemma_key_embeddings.ndim == 1:
                        vectors = lst_or_numpy_lemma_key_embeddings
                    elif lst_or_numpy_lemma_key_embeddings.ndim == 2:
                        vectors = lst_or_numpy_lemma_key_embeddings[0,:]
            else:
                if isinstance(lst_or_numpy_lemma_key_embeddings, list):
                    vectors = np.array(lst_or_numpy_lemma_key_embeddings)
                elif isinstance(lst_or_numpy_lemma_key_embeddings, np.ndarray):
                    vectors = lst_or_numpy_lemma_key_embeddings

            # normalize to unit length.
            if l2_norm:
                vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

            # adjust dimension
            if force_ndim_to_2:
                if vectors.ndim == 1:
                    # vectors: (1, n_dim)
                    vectors = vectors.reshape(1,-1)

            dict_lemma_key_embeddings[lemma_key] = vectors

        return dict_lemma_key_embeddings

    def _annotate_records(self, dict_basic_lemma_embeddings: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        lst_records = []
        for lemma_key, embeddings in dict_basic_lemma_embeddings.items():
            lemma = utils_wordnet_gloss.lemma_key_to_lemma(lemma_key)
            pos = utils_wordnet_gloss.lemma_key_to_pos(lemma_key, tagtype="short") # it returns one of ["n","v","s","r"]
            if pos not in self._target_pos:
                continue

            lemma_name = utils_wordnet_gloss.lemma_key_to_lemma_name(lemma_key)
            lemma_surface_form = utils_wordnet.lemma_to_surface_form(lemma)
            synset_id = utils_wordnet_gloss.lemma_key_to_synset_id(lemma_key)
            if self._lemma_surface_form_lowercase:
                lemma_surface_form = lemma_surface_form.lower()
            record = {
                "lemma": lemma_name,
                "lemma_surface_form": lemma_surface_form,
                "pos": pos,
                "ground_truth_lemma_keys": [lemma_key],
                "ground_truth_synset_ids": [synset_id],
                "embeddings": embeddings
            }
            lst_records.append(record)
        return lst_records

    def _reindex_dataset_using_lemma_and_pos(self, dataset):
        dict_lemma_and_pos_index = defaultdict(list)
        for idx, record in enumerate(dataset):
            tup_lemma_pos = (record["lemma"], record["pos"])
            dict_lemma_and_pos_index[tup_lemma_pos].append(idx)

        return dict_lemma_and_pos_index

    def _reindex_dataset_using_lemma_key(self, dataset):
        dict_lemma_key_index = defaultdict(list)
        for idx, record in enumerate(dataset):
            for lemma_key in record["ground_truth_lemma_keys"]:
                dict_lemma_key_index[lemma_key].append(idx)

        return dict_lemma_key_index

    def _map_lemma_and_pos_to_lemma_keys(self, dataset):
        dict_lemma_and_pos_to_lemma_keys_index = defaultdict(list)
        for record in dataset:
            tup_lemma_pos = (record["lemma"], record["pos"])
            dict_lemma_and_pos_to_lemma_keys_index[tup_lemma_pos].extend(record["ground_truth_lemma_keys"])

        return dict_lemma_and_pos_to_lemma_keys_index

    def get_records_by_lemma_key(self, lemma_key: str) -> List[Dict[str, Any]]:
        lst_records = []
        for idx in self._index_by_lemma_key[lemma_key]:
            lst_records.append(self.__getitem__(idx))
        return lst_records

    def get_records_by_lemma_and_pos(self, lemma: str, pos: str) -> List[Dict[str, Any]]:
        lst_records = []
        for idx in self._index_by_lemma_and_pos[(lemma, pos)]:
            lst_records.append(self.__getitem__(idx))
        return lst_records

    def get_lemma_and_pos(self):
        return list(set(self._index_by_lemma_and_pos.keys()))

    def get_lemmas(self, pos: str):
        lst_lemmas = [_lemma for _lemma, _pos in self._index_by_lemma_and_pos.keys() if _pos == pos]
        return list(set(lst_lemmas))

    def get_lemma_keys_by_lemma_and_pos(self, lemma: str, pos: str):
        return self._lemma_and_pos_to_lemma_keys[(lemma, pos)]

    def get_lemma_keys(self):
        return list(self._index_by_lemma_key.keys())

    @property
    def verbose(self):
        ret = {
            "target_pos": self._target_pos,
            "lemma_surface_form_lowercase": self._lemma_surface_form_lowercase,
            "path": self._path_basic_lemma_embeddings,
            "__len__": self.__len__(),
            "num_lemma_and_pos": len(self.get_lemma_and_pos()),
            "num_lemma_keys": len(self.get_lemma_keys()),
            "description":self._description
        }
        return ret

class BERTLemmaEmbeddingsDataset(SREFLemmaEmbeddingsDataset):

    _AVAILABLE_POOLING_METHOD = ("average", "cls", "entity")

    def __init__(self, kwargs_bert_embeddings_dataset: [Dict, Any],
                 pooling_method: str,
                 l2_norm: bool,
                 use_first_embeddings_only: bool = True,
                 lemma_surface_form_lowercase: bool = False,
                 target_pos: List[str] = ["n","v","s","r"],
                 description: str = ""):

        self._target_pos = target_pos
        self._lemma_surface_form_lowercase = lemma_surface_form_lowercase
        self._description = description

        assert pooling_method in self._AVAILABLE_POOLING_METHOD, f"invalid pooling_methov value: {pooling_method}"
        self._pooling_method = pooling_method

        bert_embeddings_dataset = BERTEmbeddingsDataset(**kwargs_bert_embeddings_dataset)
        self._bert_embeddings_dataset_verbose = copy.deepcopy(bert_embeddings_dataset.verbose)
        force_ndim_to_2 = False if use_first_embeddings_only else True
        dict_basic_lemma_embeddings = self.load_bert_lemma_embeddings(bert_embeddings_dataset=bert_embeddings_dataset,
                                                                      pooling_method=pooling_method, l2_norm=l2_norm,
                                                                      return_first_embeddings_only=use_first_embeddings_only,
                                                                      force_ndim_to_2=force_ndim_to_2)

        # reformat list of records. each record contains single lemma sense key and its embeddings.
        self._dataset = self._annotate_records(dict_basic_lemma_embeddings)
        self._index_by_lemma_and_pos = self._reindex_dataset_using_lemma_and_pos(dataset=self._dataset)
        self._index_by_lemma_key = self._reindex_dataset_using_lemma_key(dataset=self._dataset)
        self._lemma_and_pos_to_lemma_keys = self._map_lemma_and_pos_to_lemma_keys(dataset=self._dataset)

    @classmethod
    def AVAILABLE_POOLING_METHOD(cls):
        return cls._AVAILABLE_POOLING_METHOD

    @classmethod
    def load_bert_lemma_embeddings(cls, bert_embeddings_dataset: BERTEmbeddingsDataset,
                                   pooling_method: str, l2_norm: bool, return_first_embeddings_only: bool,
                                   force_ndim_to_2: bool = False) -> Dict[str, np.ndarray]:
        dict_lemma_key_embeddings = {}

        for obj_sentence in tqdm(bert_embeddings_dataset):
            record = obj_sentence["record"]
            v_context_embeddings = tensor_to_numpy(obj_sentence["embedding"])
            lst_lst_entity_spans = extract_entity_spans_from_record(record,
                                                                    entity_field_name="entities",
                                                                    span_field_name="subword_spans")
            lst_entity_avg_embeddings = calc_entity_subwords_average_vectors(
                                                                    context_embeddings=v_context_embeddings,
                                                                    lst_lst_entity_subword_spans=lst_lst_entity_spans)
            for entity, t_entity_avg_embedding in zip(record["entities"], lst_entity_avg_embeddings):
                if pooling_method == "cls":
                    # [cls] token is the first subword token.
                    entity_vector = v_context_embeddings[0,:]
                elif pooling_method == "average":
                    # simply take average over all subword embeddings
                    entity_vector = np.mean(v_context_embeddings, axis=0, keepdims=False)
                elif pooling_method == "entity":
                    entity_vector = tensor_to_numpy(t_entity_avg_embedding)

                # (optional) normalize to unit length.
                if l2_norm:
                    entity_vector = entity_vector / np.linalg.norm(entity_vector, axis=-1, keepdims=True)

                # (optional) adjust dimension
                if force_ndim_to_2:
                    if entity_vector.ndim == 1:
                        # vectors: (1, n_dim)
                        entity_vector = entity_vector.reshape(1,-1)

                # record as (lemma_key, embedding) pair.
                for lemma_key in entity["ground_truth_lemma_keys"]:
                    if lemma_key in dict_lemma_key_embeddings:
                        if return_first_embeddings_only:
                            pass
                        else:
                            v_orig = dict_lemma_key_embeddings[lemma_key]
                            dict_lemma_key_embeddings[lemma_key] = np.vstack([v_orig, entity_vector])
                    else:
                        dict_lemma_key_embeddings[lemma_key] = entity_vector

        return dict_lemma_key_embeddings

    @property
    def verbose(self):
        ret = {
            "bert_embeddings":self._bert_embeddings_dataset_verbose,
            "pooling_method": self._pooling_method,
            "target_pos": self._target_pos,
            "lemma_surface_form_lowercase": self._lemma_surface_form_lowercase,
            "__len__": self.__len__(),
            "num_lemma_and_pos": len(self.get_lemma_and_pos()),
            "num_lemma_keys": len(self.get_lemma_keys()),
            "description":self._description,
        }
        return ret
