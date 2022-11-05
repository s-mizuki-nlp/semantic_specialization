#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Union, List, Optional
import torch
from torch import nn
from torch.nn.modules import loss as L
from torch.nn.functional import cross_entropy
import numpy as np

from dataset.utils import numpy_to_tensor
from .similarity import CosineSimilarity
from .utils import _reduction


def _create_mask_tensor(seq_lens: Union[np.ndarray, torch.LongTensor]) -> torch.BoolTensor:
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

    def __init__(self, similarity_module: nn.Module,
                 use_positives_as_in_batch_negatives: bool = True,
                 coef_for_hard_negatives: float = 1.0,
                 size_average=None, reduce=None, reduction: str = "mean"):
        super(ContrastiveLoss, self).__init__(size_average, reduce, reduction)
        self._similarity = similarity_module
        self._use_positives_as_in_batch_negatives = use_positives_as_in_batch_negatives
        self._coef_for_hard_negatives = coef_for_hard_negatives
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction

        assert coef_for_hard_negatives > 0, f"invalid value. `coef_for_hard_negatives` must be positive."

    def get_off_diagonal_elements(self, squared_tensor):
        n = squared_tensor.shape[0]
        return squared_tensor.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)

    def forward(self, queries: torch.Tensor, positives: torch.Tensor, negatives: Optional[torch.Tensor] = None, num_negative_samples: Optional[torch.LongTensor] = None):
        # queries, positives: (n, n_dim)
        # negatives: (n, n_neg = max({n^neg_i};0<=i<n), n_dim)
        # num_negative_samples: (n,)
        # num_negative_samples[i] = [1, n_neg]; number of effective negative samples for i-th example.

        if not self._use_positives_as_in_batch_negatives:
            assert negatives is not None, f"you must feed `negatives` because `use_positives_as_in_batch_negatives=False`"

        n_batch, device = queries.shape[0], queries.device

        # vec_sim_pos: (n,)
        # is_hard_examples: affects when similarity_module is ArcMarginProduct.
        # by specifying is_hard_examples=True, pairs get even closer. i.e., we should specify True for (target, positive) pairs.
        vec_sim_pos = self._similarity(queries, positives, is_hard_examples=True)

        # in-batch negatives: (n, n-1)
        if self._use_positives_as_in_batch_negatives:
            # mat_sim: (n, n)
            mat_sim = self._similarity(queries.unsqueeze(dim=1), positives, is_hard_examples=False)
            # we extract off-diagonal elements as the in-batch negatives.
            mat_sim_neg_in_batch = self.get_off_diagonal_elements(mat_sim)
        else:
            mat_sim_neg_in_batch = None

        # (optional) explicit negatives: (n, n_neg)
        if negatives is not None:
            mat_sim_neg_explicit = self._similarity(queries.unsqueeze(dim=1), negatives, is_hard_examples=False)
            mat_sim_neg_explicit = np.log(self._coef_for_hard_negatives) + mat_sim_neg_explicit
        else:
            mat_sim_neg_explicit = None

        # concat with in-batch negatives
        if (mat_sim_neg_in_batch is not None) and (mat_sim_neg_explicit is not None):
            # mat_sim_neg: (n, n-1+n_neg)
            mat_sim_neg = torch.cat([mat_sim_neg_in_batch, mat_sim_neg_explicit], dim=-1)
            num_negative_samples = num_negative_samples + (n_batch-1)
        elif (mat_sim_neg_in_batch is not None):
            mat_sim_neg = mat_sim_neg_in_batch
            num_negative_samples = torch.tensor([n_batch-1]*n_batch, device=device)
        elif (mat_sim_neg_explicit is not None):
            mat_sim_neg = mat_sim_neg_explicit

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


class TripletLoss(ContrastiveLoss):

    def __init__(self, margin: float = 0.0,
                 use_positives_as_in_batch_negatives: bool = True,
                 size_average=None, reduce=None, reduction: str = "mean"):
        similarity_module = CosineSimilarity(temperature=1.0)
        super(TripletLoss, self).__init__(similarity_module, use_positives_as_in_batch_negatives, size_average, reduce, reduction)
        self._margin = margin

    def forward(self, queries: torch.Tensor, positives: torch.Tensor, negatives: Optional[torch.Tensor] = None, num_negative_samples: Optional[torch.LongTensor] = None):
        # queries, positives: (n, n_dim)
        # negatives: (n, n_neg = max({n^neg_i};0<=i<n), n_dim)
        # num_negative_samples: (n,)
        # num_negative_samples[i] = [1, n_neg]; number of effective negative samples for i-th example.

        if not self._use_positives_as_in_batch_negatives:
            assert negatives is not None, f"you must feed `negatives` because `use_positives_as_in_batch_negatives=False`"

        n_batch, device = queries.shape[0], queries.device

        # vec_sim_pos: (n,)
        # is_hard_examples: affects when similarity_module is ArcMarginProduct.
        # by specifying is_hard_examples=True, pairs get even closer. i.e., we should specify True for (target, positive) pairs.
        vec_sim_pos = self._similarity(queries, positives, is_hard_examples=True)

        # in-batch negatives: (n, n-1)
        if self._use_positives_as_in_batch_negatives:
            # mat_sim: (n, n)
            mat_sim = self._similarity(queries.unsqueeze(dim=1), positives, is_hard_examples=False)
            # we extract off-diagonal elements as the in-batch negatives.
            mat_sim_neg_in_batch = self.get_off_diagonal_elements(mat_sim)
        else:
            mat_sim_neg_in_batch = None

        # (optional) explicit negatives: (n, n_neg)
        if negatives is not None:
            mat_sim_neg_explicit = self._similarity(queries.unsqueeze(dim=1), negatives, is_hard_examples=False)
        else:
            mat_sim_neg_explicit = None

        # concat with in-batch negatives
        if (mat_sim_neg_in_batch is not None) and (mat_sim_neg_explicit is not None):
            # mat_sim_neg: (n, n-1+n_neg)
            mat_sim_neg = torch.cat([mat_sim_neg_in_batch, mat_sim_neg_explicit], dim=-1)
            num_negative_samples = num_negative_samples + (n_batch-1)
        elif (mat_sim_neg_in_batch is not None):
            mat_sim_neg = mat_sim_neg_in_batch
            num_negative_samples = torch.tensor([n_batch-1]*n_batch, device=device)
        elif (mat_sim_neg_explicit is not None):
            mat_sim_neg = mat_sim_neg_explicit

        # fill -inf with masked positions
        # mask_tensor: (n, max(num_negative_samples)) = {True for valid positions, False for invalid positions}
        mask_tensor = _create_mask_tensor(num_negative_samples)
        mask_tensor = torch.logical_not(mask_tensor)

        # calculate triplet loss
        # mat_triplet_loss: (n, max(num_negative_samples))
        # mat_triplet_loss[i,j] = max( sim(i, i_neg_j) - sim(i, i_pos) + margin, 0 )
        mat_triplet_loss = mat_sim_neg - vec_sim_pos.unsqueeze(dim=-1) + self._margin
        mat_triplet_loss = torch.maximum(mat_triplet_loss, torch.zeros_like(mat_sim_neg)) * mask_tensor
        # average over negatives
        losses = torch.sum(mat_triplet_loss, dim=-1, keepdim=False) / torch.sum(mask_tensor, dim=-1, keepdim=False)

        return _reduction(losses, self.reduction)