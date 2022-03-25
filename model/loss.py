#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Union, List
import torch
from torch import nn
from torch.nn.modules import loss as L
from torch.nn.functional import cross_entropy
import numpy as np

from dataset.utils import numpy_to_tensor

def _create_mask_tensor(seq_lens: Union[np.ndarray, torch.LongTensor]):
    seq_lens = numpy_to_tensor(seq_lens)
    max_len = torch.max(seq_lens)

    # create tensor of suitable shape and same number of dimensions
    range_tensor = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
    range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))

    # until this step, we only created auxiliary tensors (you may already have from previous steps)
    # the real mask tensor is created with binary masking:
    mask_tensor = (range_tensor >= seq_lens.unsqueeze(1))

    return mask_tensor


class ContrastiveLoss(L._Loss):

    def __init__(self, similarity_module: nn.Module, size_average=None, reduce=None, reduction: str = "mean"):
        super(ContrastiveLoss, self).__init__(size_average, reduce, reduction)
        self._similarity = similarity_module
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction

    def forward(self, queries: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, num_negative_samples: torch.LongTensor):
        # queries, positives: (n, n_dim)
        # negatives: (n, n_neg = max({n^neg_i};0<=i<n), n_dim)
        # num_negative_samples: (n,)
        # num_negative_samples[i] = [1, n_neg]; number of effective negative samples for i-th example.

        # vec_sim_pos: (n,)
        # is_hard_examples: affects when similarity_module is ArcMarginProduct.
        vec_sim_pos = self._similarity(queries, positives, is_hard_examples = True)
        # mat_sim_neg: (n, n_neg)
        mat_sim_neg = self._similarity(queries.unsqueeze(dim=1), negatives, is_hard_examples=False)
        # fill -inf with masked positions
        mask_tensor = _create_mask_tensor(num_negative_samples)
        mat_sim_neg = mat_sim_neg.masked_fill_(mask_tensor, value=-float("inf"))
        # mat_sim: (n, 1 + n_neg); positive pseudo logit score: mat_sim[:,0]
        mat_sim = torch.cat([vec_sim_pos.unsqueeze(dim=-1), mat_sim_neg], dim=-1)

        # log likelihood of positive examples
        # targets: (n,). targets[i] = 0; ground-truth (=positive) class index is always zero.
        targets = torch.zeros_like(vec_sim_pos, dtype=torch.long)
        losses = cross_entropy(input=mat_sim, target=targets, size_average=self._size_average, reduce=self._reduce, reduction=self._reduction)

        return losses