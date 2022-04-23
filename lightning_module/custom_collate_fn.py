#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, List, Dict, Any, Union
import torch
import pydash
from dataset import utils

class ContrastiveDatasetCollateFunction(object):

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
