#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Set, Optional, Dict, Any, Iterator, Union, List, Iterable, Callable

import torch
import numpy as np

from .contextualized_embeddings import BERTEmbeddingsDataset

from torch.utils.data import IterableDataset
from .encoder import extract_entity_subword_embeddings, calc_entity_subwords_average_vectors, extract_entity_spans_from_record
from . import utils
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss

class WSDTaskDataset(IterableDataset):

    def __init__(self, has_ground_truth: bool,
                 bert_embeddings_dataset: BERTEmbeddingsDataset,
                 return_level: str = "entity",
                 record_entity_field_name: str = "entities",
                 record_entity_span_field_name: str = "subword_spans",
                 ground_truth_lemma_keys_field_name: Optional[str] = None,
                 copy_field_names_from_record_to_entity: Optional[Iterable[str]] = None,
                 return_entity_subwords_avg_vector: bool = True,
                 weighted_average_entity_embeddings_and_sentence_embedding: float = 0.0,
                 normalize_embeddings: bool = False,
                 excludes: Optional[Set[str]] = None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None):

        self._has_ground_truth = has_ground_truth
        self._bert_embeddings = bert_embeddings_dataset
        self._return_level = return_level
        self._record_entity_field_name = record_entity_field_name
        self._record_entity_span_field_name = record_entity_span_field_name
        self._ground_truth_lemma_keys_field_name = ground_truth_lemma_keys_field_name
        self._copy_field_names_from_record_to_entity = copy_field_names_from_record_to_entity
        self._return_entity_subwords_avg_vector = return_entity_subwords_avg_vector
        self._normalize_embeddings = normalize_embeddings
        self._weighted_average_entity_embeddings_and_sentence_embedding = weighted_average_entity_embeddings_and_sentence_embedding
        self._excludes = set() if excludes is None else excludes

        if filter_function is None:
            self._filter_function = []
        elif isinstance(filter_function, list):
            self._filter_function = filter_function
        elif not isinstance(filter_function, list):
            self._filter_function = [filter_function]

    def _copy_fields(self, dict_source: Dict[str, Any], dict_target: Dict[str, Any],
                     copy_field_names: Optional[Iterable[str]] = None):
        if copy_field_names is None:
            return dict_target

        for field_name in copy_field_names:
            dict_target[field_name] = dict_source[field_name]
        return dict_target

    def _entity_loader(self) -> Iterator[Dict[str, Any]]:
        for obj_sentence in self._sentence_loader():
            for obj_entity in self._yield_entities_from_sentence(obj_sentence, include_embeddings=True):
                yield obj_entity

    def _yield_entities_from_sentence(self, obj_sentence, include_embeddings: bool = True):
        obj_sentence_record = obj_sentence["record"]
        lst_entities = obj_sentence_record[self._record_entity_field_name]

        if include_embeddings:
            lst_entity_embeddings = obj_sentence["entity_embeddings"]
            lst_entity_seq_len = obj_sentence["entity_sequence_lengths"]
            lst_entity_span_avg_vectors = obj_sentence.get("entity_span_avg_vectors", [])
            context_embedding = obj_sentence["embedding"]
            context_sequence_length = obj_sentence["sequence_length"]

        for idx, dict_entity in enumerate(lst_entities):
            dict_entity = self._copy_fields(dict_source=obj_sentence_record, dict_target=dict_entity,
                                            copy_field_names=self._copy_field_names_from_record_to_entity)

            lemma, pos = dict_entity["lemma"], dict_entity["pos"]

            obj_entity = {
                "lemma":lemma,
                "pos":pos
            }
            if include_embeddings:
                embeddings = {
                    "entity_embedding": lst_entity_embeddings[idx],
                    "entity_sequence_length": lst_entity_seq_len[idx],
                    "context_embedding": context_embedding,
                    "context_sequence_length": context_sequence_length,
                }
                # (optional) compute average vector of entity spans (subword-level average, then word-level average)
                if self._return_entity_subwords_avg_vector:
                    embeddings["entity_span_avg_vector"] = lst_entity_span_avg_vectors[idx]
                obj_entity.update(embeddings)

            # assign ground-truth synset
            if self._has_ground_truth: # training dataset
                lemma_keys = dict_entity[self._ground_truth_lemma_keys_field_name]
                synset_ids = list(map(utils_wordnet_gloss.lemma_key_to_synset_id, lemma_keys))
                lexnames = list(map(utils_wordnet_gloss.lemma_key_to_lexname, lemma_keys))

                obj_entity["ground_truth_synset_id"] = synset_ids[0]
                obj_entity["ground_truth_lexname"] = lexnames[0]

            else: # evaluation dataset -> dataset.evalution.WSDEvaluationDataset
                pass

            obj_entity.update(dict_entity)

            yield obj_entity

    def _weighed_average_on_entity_embeddings(self, entity_embeddings: List[np.ndarray],
                                              context_embeddings: np.ndarray, weight_alpha: float) -> List[np.ndarray]:
        """
        calculate weighted average over entity embeddings and sentence embedding (=simple average over context embeddings)

        @param entity_embeddings: list of the sequence of subword embeddings of the entities. shape: List[(n_window, n_dim)]
        @param context_embeddings: sequence of subword embeddings of a sentence. shape: (n_seq_len, n_dim)
        @param weight_alpha: weight parameter. result = (1.0 - weight_alpha) * entity_embedding + weight_alpha * sentence_embedding
        @return:
        """
        if weight_alpha == 0.0:
            return entity_embeddings

        lst_new_embeddings = []
        # sentence_embedding: (n_dim,)
        sentence_embedding = context_embeddings.mean(axis=0)
        for entity_embedding in entity_embeddings:
            if weight_alpha == 1.0:
                # completely replace all subword embs with sentence embedding.
                n_window = entity_embedding.shape[0]
                new_embedding = np.repeat(sentence_embedding.reshape(1,-1), n_window, axis=0)
            else:
                new_embedding = (1.0 - weight_alpha) * entity_embedding + weight_alpha * sentence_embedding
            lst_new_embeddings.append(new_embedding)

        return lst_new_embeddings

    def _sentence_loader(self) -> Iterator[Dict[str, Any]]:
        """
        returns sentence-level objects.

        returns:
            embedding: sequence of subword embeddings of a sentence. shape: (n_seq_len, n_dim)
            sequence_length: number of subwords in a sentence.
            record: sentence information.
            entity_embeddings: list of the sequence of subword embeddings of the entities. shape: List[(n_window, n_dim)]
            entity_subword_lengths: list of the entity subword window sizes. List[n_window]
        """
        for obj_sentence in self._bert_embeddings:
            record = obj_sentence["record"]
            lst_lst_entity_spans = extract_entity_spans_from_record(record,
                                                                    entity_field_name=self._record_entity_field_name,
                                                                    span_field_name=self._record_entity_span_field_name)
            # keys: embeddings, sequence_lengths
            dict_entity_embeddings = extract_entity_subword_embeddings(
                                     context_embeddings=obj_sentence["embedding"],
                                     lst_lst_entity_subword_spans=lst_lst_entity_spans,
                                     padding=False)
            obj_sentence["entity_embeddings"] = dict_entity_embeddings["embeddings"]
            obj_sentence["entity_sequence_lengths"] = dict_entity_embeddings["sequence_lengths"]

            if self._return_entity_subwords_avg_vector:
                if "entity_span_avg_vectors" in obj_sentence:
                    pass
                elif "embedding" in obj_sentence:
                    # compute entity span average vectors on-the-fly.
                    obj_sentence["entity_span_avg_vectors"] = calc_entity_subwords_average_vectors(
                                                                context_embeddings=obj_sentence["embedding"],
                                                                lst_lst_entity_subword_spans=lst_lst_entity_spans)
                else:
                    raise AttributeError(f"neither `entity_span_avg_vectors` nor `embedding` in the record.")
            # (optional) weighted average over entity embeddings and sentence embedding
            obj_sentence["entity_embeddings"] = self._weighed_average_on_entity_embeddings(
                                                            entity_embeddings=obj_sentence["entity_embeddings"],
                                                            context_embeddings=obj_sentence["embedding"],
                                                            weight_alpha=self._weighted_average_entity_embeddings_and_sentence_embedding
                                                        )
            # (optional) normalize embeddings
            if self._normalize_embeddings:
                obj_sentence["entity_embeddings"] = list(map(utils.l2_norm, obj_sentence["entity_embeddings"]))
                obj_sentence["embedding"] = utils.l2_norm(obj_sentence["embedding"])
                if self._return_entity_subwords_avg_vector:
                    obj_sentence["entity_span_avg_vectors"] = list(map(utils.l2_norm, obj_sentence["entity_span_avg_vectors"]))

            yield obj_sentence

    def __iter__(self):
        if self._return_level == "entity":
            it_records = self._entity_loader()
        elif self._return_level == "sentence":
            it_records = self._sentence_loader()
        else:
            raise ValueError(f"unknown `return_level` value: {self._return_level}")
        for record in it_records:
            if self._filter(record) == True:
                continue
            for exclude_field in self._excludes:
                _ = record.pop(exclude_field, None)
            yield record

    def __len__(self):
        if hasattr(self, "_n_records"):
            return self._n_records
        else:
            self._n_records = self.count_records()
        return self._n_records

    def _filter(self, entry: Dict[str, Any]):
        for filter_function in self._filter_function:
            if filter_function(entry) == True:
                return True
        return False

    def count_records(self):
        n_records = 0
        for _ in self.record_loader(return_level=self._return_level):
            n_records += 1
        return n_records

    @property
    def embeddings_dataset(self):
        return self._bert_embeddings

    @property
    def has_ground_truth(self):
        return self._has_ground_truth

    def record_loader(self, return_level: str):
        self._bert_embeddings.return_record_only = True

        for obj_sentence in self._bert_embeddings:
            if return_level == "entity":
                for obj_entity in self._yield_entities_from_sentence(obj_sentence, include_embeddings=False):
                    if self._filter(obj_entity) == True:
                        continue
                    yield obj_entity

            elif return_level == "sentence":
                if self._filter(obj_entity) == True:
                        continue
                yield obj_sentence

        self._bert_embeddings.return_record_only = False

    @property
    def verbose(self):
        lst_attr_names = "has_ground_truth,return_level".split(",")
        ret = {attr_name:getattr(self, "_" + attr_name) for attr_name in lst_attr_names}
        ret["__len__"] = self.__len__()
        ret["bert_embeddings_dataset"] = self._bert_embeddings.verbose
        return ret


class WSDTaskDatasetCollateFunction(object):

    def __init__(self,
                 has_ground_truth: bool,
                 return_records: bool = True,
                 return_entity_context_attn_mask: bool = False,
                 num_heads_entity_context_mha: Optional[int] = None,
                 device: Optional[str] = "cpu"):

        self._has_ground_truth = has_ground_truth
        self._return_records = return_records
        self._return_entity_context_attn_mask = return_entity_context_attn_mask

        if return_entity_context_attn_mask:
            assert isinstance(num_heads_entity_context_mha, int), \
                f"you must specify the number of attention heads of MHA module as: `num_heads_entity_context_mha`"
        self._num_heads = num_heads_entity_context_mha
        self._device = device

    def __call__(self, lst_entity_objects: List[Dict[str, Any]]):
        def _list_of(field_name: str):
            return [obj[field_name] for obj in lst_entity_objects]

        set_field_names = next(iter(lst_entity_objects)).keys()

        # token info, context embeddings and entity embeddings
        lst_lemmas = _list_of("lemma")
        lst_pos = _list_of("pos")
        lst_subword_spans = _list_of("subword_spans")
        lst_context_sequence_lengths = _list_of("context_sequence_length")
        lst_lagged_context_embeddings = _list_of("context_embedding")
        lst_entity_sequence_lengths = _list_of("entity_sequence_length")
        lst_lagged_entity_span_embeddings = _list_of("entity_embedding")
        dict_ret = {
            "lemmas": lst_lemmas,
            "pos": lst_pos,
            "subword_spans": lst_subword_spans,
            "context_sequence_lengths": torch.tensor(lst_context_sequence_lengths).to(self._device),
            "context_embeddings": utils.pad_and_stack_list_of_tensors(lst_lagged_context_embeddings).to(self._device),
            "entity_sequence_lengths": torch.tensor(lst_entity_sequence_lengths).to(self._device),
            "entity_embeddings": utils.pad_and_stack_list_of_tensors(lst_lagged_entity_span_embeddings).to(self._device)
        }
        ## (optional) entity span average vectors
        if "entity_span_avg_vector" in set_field_names:
            dict_ret["entity_span_avg_vectors"] = torch.stack(_list_of("entity_span_avg_vector")).to(self._device)

        # attention masks used for MultiheadAttention and GlobalAttention module.
        _, device = utils.get_dtype_and_device(dict_ret["context_embeddings"])
        dict_ret["entity_sequence_mask"] = utils.create_sequence_mask(lst_entity_sequence_lengths, device=device)
        dict_ret["context_sequence_mask"] = utils.create_sequence_mask(lst_context_sequence_lengths, device=device)

        ## (optional) attn_mask for MultiheadAttention module.
        if self._return_entity_context_attn_mask:
            entity_context_attn_mask = utils.create_multiheadattention_attn_mask_batch(
                lst_query_sequence_lengths=lst_entity_sequence_lengths,
                lst_key_value_sequence_lengths=lst_context_sequence_lengths,
                target_sequence_length=max(lst_entity_sequence_lengths),
                source_sequence_length=max(lst_context_sequence_lengths),
                num_heads=self._num_heads,
                device=device
            )
            dict_ret["entity_context_attn_mask"] = entity_context_attn_mask

        if self._has_ground_truth:
            # ground truth: synset code
            dict_ret["ground_truth_synset_ids"] = _list_of("ground_truth_synset_id")

        # other attributes are accumulated as `records` object.
        if self._return_records:
            trim_plural = lambda name: name[:-1] if name.endswith("s") else name
            set_essential_fields = {"pos", "lemma", "subword_spans"}
            set_caught_fields = set([trim_plural(name) for name in dict_ret.keys()])
            set_uncaught_fields = (set_field_names - set_caught_fields) | set_essential_fields
            lst_records = [{name:e_object.get(name, None) for name in set_uncaught_fields} for e_object in lst_entity_objects]
            dict_ret["records"] = lst_records

        return dict_ret