#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Union, Tuple
import torch
from torch.nn.modules import loss as L, PairwiseDistance

from .similarity import CosineSimilarity
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

    def __init__(self, similarity_module: Optional[torch.nn.Module] = None,
                 label_threshold: float = 0.0, top_k: int = 1,
                 repel_top_k: Optional[int] = None,
                 size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)

        self._similarity = CosineSimilarity(temperature=1.0) if similarity_module is None else similarity_module
        self._label_threshold = label_threshold
        self._top_k = top_k
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction
        if repel_top_k is not None:
            # if int & greater than top_k, it takes rest of top-k as negatives.
            if repel_top_k > 0:
                assert repel_top_k > top_k, f"`repel_top_k` must be greater than `top_k` value when it is positive integer."
                self._repel_top_k = repel_top_k
            else:
                warnings.warn(f"we will take the least top-k examples as repelling: {repel_top_k}")
                self._repel_top_k = repel_top_k
        else:
            self._repel_top_k = None

    def _forward_repel(self, mat_sim: torch.Tensor, mask_tensor: torch.Tensor, attract_top_k: int, repel_top_k: int) -> Tuple[torch.Tensor, int]:

        # sort by descending order. invalid elements will be trailed with -inf.
        mat_sim_sorted, _ = torch.sort(mat_sim.masked_fill(mask_tensor, value=-float("inf")), dim=-1, descending=True)

        if repel_top_k > 0:
            # take most similar repel_top_k examples excluding most similar topk eamples.

            ## 1. take most similar non-topk examples
            mat_sim_topk_negatives = mat_sim_sorted[:, attract_top_k:repel_top_k]

            ## 2. count number of targets.
            t_denom = (mat_sim_topk_negatives != -float("inf")).sum(dim=-1)

            ## 3. take average excluding invalid elements (= inf elements)
            losses = mat_sim_topk_negatives.nan_to_num(neginf=0.0).sum(dim=-1) / t_denom.clip(min=1.0)
            n_samples = max(1, (t_denom > 0).sum().item())

        else:
            # take least similar topk examples excluding most similar topk eamples.

            ## 1. exclude most similar topk.
            mat_sim_non_topk = mat_sim_sorted[:, attract_top_k:]

            ## 2. take least similar topk. invalid elements are flipped to inf in order to sort ascending order.
            mat_sim_least_non_topk = \
            torch.sort(mat_sim_non_topk.nan_to_num(neginf=float("inf")), dim=-1, descending=False)[0][:, :(-repel_top_k)]

            ## 3. take average excluding invalid elements (= inf elements)
            t_denom = (mat_sim_least_non_topk != float("inf")).sum(dim=-1)
            losses = mat_sim_least_non_topk.nan_to_num(posinf=0.0).sum(dim=-1) / t_denom.clip(min=1.0)
            n_samples = max(1, (t_denom > 0).sum().item())

        # losses will be zeroes if there is no valid elements.
        return losses, n_samples

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
        mat_sim_pos = mat_sim.masked_fill(mask_tensor, value=-float("inf"))
        # vec_sim_topk: (n,); top-k average similarity for each query.
        if self._top_k == 1:
            vec_sim_topk, _ = torch.max(mat_sim_pos, dim=-1)
        else:
            mat_sim_topk, _ = torch.topk(mat_sim_pos, k=self._top_k)
            # replace invalid elements with zeroes
            mat_sim_topk = mat_sim_topk.masked_fill(mask_tensor[:, :self._top_k], value=0.0)
            # take top-k average while number of target samples into account.
            t_denom = self._top_k - mask_tensor[:, :self._top_k].sum(dim=-1)
            vec_sim_topk = mat_sim_topk.sum(dim=-1) / t_denom

        # compare the threshold with similarity diff. between top-1 and top-2.
        # dummy elements are masked by -inf, then it naturally exceeds threshold = regarded as valid example.
        if mat_sim.shape[-1] > 1:
            obj = torch.topk(mat_sim_pos, k=2, largest=True)
            is_valid_sample = (obj.values[:,0] - obj.values[:,1]) > self._label_threshold
        else:
            is_valid_sample = torch.ones_like(vec_sim_topk).type(torch.bool)

        # loss = 1.0 - negative top-k similarity as long as
        losses = (1.0 - vec_sim_topk) * is_valid_sample + 1.0 * (is_valid_sample == False)
        n_samples = max(1, is_valid_sample.sum().item())

        # take non-topk examples as negatives to repel them.
        # loss = mean_{s' \in targets[rng_negative_top_k]}(\rho_{s,s'})
        if isinstance(self._repel_top_k, int):
            losses_repel, n_samples_repel = self._forward_repel(mat_sim=mat_sim, mask_tensor=mask_tensor,
                                                                attract_top_k=self._top_k, repel_top_k=self._repel_top_k)
        else:
            losses_repel = None
            n_samples_repel = None

        ret_loss = _reduction(losses=losses, reduction=self.reduction, num_samples=n_samples)
        if losses_repel is not None:
            ret_loss = ret_loss + _reduction(losses=losses_repel, reduction=self.reduction, num_samples=n_samples_repel)

        return ret_loss