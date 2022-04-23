#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, List, Dict, Any, Union
import torch
import pydash
from dataset import utils

from dataset.gloss_embeddings import SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset

class ContrastiveDatasetEmbeddingsCollateFunction(object):

    def __init__(self,
                 device: Optional[str] = "cpu"):
        self._device = device

    def __call__(self, lst_records: List[Dict[str, Any]]) -> Dict[str, Union[None, torch.Tensor]]:
        """

        Args:
            lst_records: list of outputs from ContrastiveLearningDataset.

        Returns: Dict[str, torch.tensor]
            query: (n, n_dim), query embeddings
            positive: (n, n_dim), positive entity embeddings
            num_hard_negatives: (n,) or None, number of hard negative examples.
            hard_negatives: (n, max(num_hard_negatives), n_dim) or None, hard negative entity embeddings.
        """

        def _list_of(field_name: str, lst_records):
            return [pydash.get(obj, field_name) for obj in lst_records]

        lst_queries = _list_of("query.embeddings", lst_records)
        lst_positives = _list_of("positive.embeddings", lst_records)
        lst_num_hard_negatives = _list_of("num_hard_negatives", lst_records)
        n, n_dim = len(lst_queries), len(lst_queries[0])
        # if all values are zeroes, then we regard hard negative examples are not available.
        is_hard_negative_exists = lst_num_hard_negatives.count(0) != n

        # hard negatives
        if is_hard_negative_exists:
            lst_t_hard_negatives = []
            dummy = torch.zeros((1, n_dim), dtype=torch.float)
            for record in lst_records:
                if record["num_hard_negatives"] == 0:
                    lst_t_hard_negatives.append(dummy)
                else:
                    lst_hard_negatives = _list_of("embeddings", record["hard_negatives"])
                    lst_t_hard_negatives.append(torch.tensor(lst_hard_negatives))
            t_hard_negatives = utils.pad_and_stack_list_of_tensors(lst_t_hard_negatives).to(self._device)
        else:
            t_hard_negatives = None

        dict_ret = {
            "query": torch.tensor(lst_queries, device=self._device),
            "positive": torch.tensor(lst_positives, device=self._device),
            "num_hard_negatives": torch.LongTensor(lst_num_hard_negatives, device=self._device) if is_hard_negative_exists else None,
            "hard_negatives": t_hard_negatives
        }

        return dict_ret


class GlossContextSimilarityTaskEmbeddingsCollateFunction(object):

    def __init__(self,
                 lemma_embeddings_dataset: Union[SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset],
                 convert_adjective_to_adjective_satellite: bool = True,
                 device: Optional[str] = "cpu"):

        if not convert_adjective_to_adjective_satellite:
            warnings.warn(f"We recommend you to enable convert_adjective_to_adjective_satellite unless otherwise specific reason.")

        self._device = device
        self._convert_adjective_to_adjective_satellite = convert_adjective_to_adjective_satellite
        self._lemma_embeddings_dataset = lemma_embeddings_dataset

    def __call__(self, lst_records: List[Dict[str, Any]]) -> Dict[str, Union[None, torch.Tensor]]:
        """

        Args:
            lst_records: list of outputs from WSDTaskDataset.

        Returns: Dict[str, torch.tensor]
            query: (n, n_dim), query embeddings
            positive: (n, n_dim), positive entity embeddings
            num_hard_negatives: (n,) or None, number of hard negative examples.
            hard_negatives: (n, max(num_hard_negatives), n_dim) or None, hard negative entity embeddings.
        """

        def _list_of(field_name: str, lst_records):
            return [pydash.get(obj, field_name) for obj in lst_records]

        lst_t_queries = []
        lst_t_targets = []
        lst_num_targets = []
        for record in lst_records:
            lemma_name, pos = record["lemma"], record["pos"]
            if self._convert_adjective_to_adjective_satellite:
                pos = "s" if pos == "a" else pos

            assert record["entity_sequence_length"] == record["entity_embedding"].shape[0]
            # query
            t_query = torch.mean(record["entity_embedding"], dim=0)
            lst_t_queries.append(t_query)

            # targets = candidate glosses
            lst_target_records = self._lemma_embeddings_dataset.get_records_by_lemma_and_pos(lemma=lemma_name, pos=pos)
            assert len(lst_target_records) > 0, f"not in lemma dataset: {lemma_name}|{pos}"
            t_target = torch.tensor(_list_of("embeddings", lst_target_records))
            lst_t_targets.append(t_target)

            # number of targets
            lst_num_targets.append(len(lst_target_records))

        t_query = torch.stack(lst_t_queries, dim=0).to(self._device)
        t_targets = utils.pad_and_stack_list_of_tensors(lst_t_targets).to(self._device)

        dict_ret = {
            "query": t_query,
            "targets": t_targets,
            "num_targets": torch.LongTensor(lst_num_targets, device=self._device)
        }

        return dict_ret