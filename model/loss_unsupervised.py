#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn.modules import loss as L, PairwiseDistance
from torch.nn.functional import pdist, normalize

from .loss import _create_mask_tensor


### unsupervised loss classes ###

class ReprPreservationLoss(L._Loss):

    def __init__(self, ord=2.0, size_average=None, reduce=None, reduction: str = "mean"):
        """
        Computes the p-norm between original repr. and transformed repr.

        Args:
            ord: order of norm.
            reduction: reduction method.
        """
        super().__init__(size_average, reduce, reduction)
        self._p_dist = PairwiseDistance(p=ord)

    def forward(self, original: torch.Tensor, transformed: torch.Tensor):
        # original, transformed: (n, n_dim)
        losses = self._p_dist(original, transformed)

        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses


class PairwiseSimilarityPreservationLoss(L._Loss):

    def __init__(self, similarity: str = "cosine", size_average=None, reduce=None, reduction: str = "mean"):
        """
        Computes the mean-squared-error of the pairwise similarities between original space and transformed space.
        pairiwise similarities are the cosine/dot similarity between every pair of row vectors in the inputs.

        Args:
            similarity: similarity metric. cosine or dot.
            reduction: reduction method.
        """
        super().__init__(size_average, reduce, reduction)
        assert similarity in ("cosine","dot"), "valid `similarity` values are: {cosine,dot}"
        self._similarity = similarity

    def _pairwise_cosine_similarity(self, tensor: torch.Tensor):
        t_x = normalize(tensor, p=2.0, dim=-1)
        t_pairwise_sims = 1.0 - pdist(t_x, p=2.0) / 2
        return t_pairwise_sims

    def _pairwise_dot_similarity(self, tensor: torch.Tensor):
        t_x = tensor
        t_pairwise_l2 = torch.nn.functional.pdist(t_x)

        n_ = t_x.shape[0]
        t_norm = torch.linalg.norm(t_x, dim=-1)
        t_norm_cross = torch.tile(t_norm, (n_, 1))

        triu_index = [torch.triu_indices(n_, n_, offset=1)[0], torch.triu_indices(n_, n_, offset=1)[1]]
        t_norm_cols = t_norm_cross.T[triu_index]
        t_norm_rows = t_norm_cross[triu_index]

        t_pairwise_dot = (t_norm_cols**2 + t_norm_rows**2 - t_pairwise_l2**2) / 2
        return t_pairwise_dot

    def forward(self, original: torch.Tensor, transformed: torch.Tensor):
        # original, transformed: (n, n_dim)
        if self._similarity == "cosine":
            t_sim_orig = self._pairwise_cosine_similarity(original)
            t_sim_trans = self._pairwise_dot_similarity(transformed)
        elif self._similarity == "dot":
            t_sim_orig = self._pairwise_dot_similarity(original)
            t_sim_trans = self._pairwise_dot_similarity(transformed)

        losses = (t_sim_orig - t_sim_trans)**2

        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses


class MaxPoolingMarginLoss(L._Loss):

    def __init__(self, similarity_module: torch.nn.Module,
                 max_margin: float,
                 size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self._similarity = similarity_module
        self._max_margin = max_margin
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction

    def forward(self, queries: torch.Tensor, targets: torch.Tensor, num_target_samples: torch.LongTensor):
        # queries: (n, n_dim)
        # targetss: (n, n_tgt = max({n^tgt_i};0<=i<n), n_dim)
        # num_target_samples: (n,)
        # num_target_samples[i] = [1, n_tgt]; number of effective target samples for i-th query.

        # is_hard_examples: affects when similarity_module is ArcMarginProduct.
        # mat_sim_neg: (n, n_tgt)
        mat_sim = self._similarity(queries.unsqueeze(dim=1), targets, is_hard_examples=False)
        # fill -inf with masked positions
        mask_tensor = _create_mask_tensor(num_target_samples)
        mat_sim = mat_sim.masked_fill_(mask_tensor, value=-float("inf"))
        # vec_sim_max: (n,); maximum similarity for each query.
        vec_sim_max, _ = torch.max(mat_sim, dim=-1)

        # hinge loss
        # targets: (n,). targets[i] = 0; ground-truth (=positive) class index is always zero.
        losses = torch.maximum(torch.zeros_like(vec_sim_max), self._max_margin - vec_sim_max)

        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses
