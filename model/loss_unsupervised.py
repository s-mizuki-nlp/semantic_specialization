#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn.modules import loss as L, PairwiseDistance

from .loss import _create_mask_tensor
from .utils import pairwise_cosine_similarity, pairwise_dot_similarity, _reduction

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

        return _reduction(losses, self.reduction)


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

    def forward(self, original: torch.Tensor, transformed: torch.Tensor):
        # original, transformed: (n, n_dim)
        if self._similarity == "cosine":
            t_sim_orig = pairwise_cosine_similarity(original, reduction="none")
            t_sim_trans = pairwise_cosine_similarity(transformed, reduction="none")
        elif self._similarity == "dot":
            t_sim_orig = pairwise_dot_similarity(original, reduction="none")
            t_sim_trans = pairwise_dot_similarity(transformed, reduction="none")

        losses = (t_sim_orig - t_sim_trans)**2

        return _reduction(losses, self.reduction)


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

        return _reduction(losses, self.reduction)
