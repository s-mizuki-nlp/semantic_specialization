#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional
import torch
from torch.nn.functional import pdist, normalize

def masked_average(embeddings: torch.Tensor, sequence_mask: torch.BoolTensor, dim: int = -2):
    """
    apply masked average.

    @param embeddings: 2D or 3D embeddings with second last axes is the sequence dimension.
            shape: (n_batch, n_seq, n_dim) or (n_seq, n_dim)
    @param sequence_mask: 1D or 2D tensor where invalid sequence element value is True.
            shape: (n_batch, n_seq) or (n_seq,)
    @param dim: sequence dimension. DEFAULT: -2
    @rtype: averaged embedding along with sequence dimension. shape: (n_batch, n_dim) or (n_dim,)
    """
    assert embeddings.ndim == sequence_mask.ndim + 1, f"embeddings and mask dimension mismatch."

    # (n_batch, n_seq) -> (n_batch, n_seq, 1)
    mask_rev = ~sequence_mask.unsqueeze(dim=-1)
    # (n_batch, n_dim) / (n_batch, 1)
    t_mean = (embeddings * mask_rev).nansum(dim=dim) / (mask_rev.sum(dim=dim))

    return t_mean

def _reduction(losses: torch.Tensor, reduction: str, num_samples: Optional[int] = None):
    if reduction == "mean":
        if num_samples is None:
            return torch.mean(losses)
        else:
            return torch.sum(losses) / num_samples
    elif reduction == "sum":
        return torch.sum(losses)
    elif reduction == "none":
        return losses

def pairwise_cosine_similarity(tensor: torch.Tensor, reduction="none"):
    t_x = normalize(tensor, p=2.0, dim=-1)
    t_pairwise_sims = 1.0 - pdist(t_x, p=2.0) / 2
    return _reduction(t_pairwise_sims, reduction)

def pairwise_dot_similarity(tensor: torch.Tensor, reduction="none"):
    t_x = tensor
    t_pairwise_l2 = torch.nn.functional.pdist(t_x)

    n_ = t_x.shape[0]
    t_norm = torch.linalg.norm(t_x, dim=-1)
    t_norm_cross = torch.tile(t_norm, (n_, 1))

    triu_index = [torch.triu_indices(n_, n_, offset=1)[0], torch.triu_indices(n_, n_, offset=1)[1]]
    t_norm_cols = t_norm_cross.T[triu_index]
    t_norm_rows = t_norm_cross[triu_index]

    t_pairwise_dot = (t_norm_cols**2 + t_norm_rows**2 - t_pairwise_l2**2) / 2
    return _reduction(t_pairwise_dot, reduction)

def batch_pairwise_cosine_similarity(tensors: torch.Tensor, num_samples: torch.LongTensor, reduction="none"):
    losses = torch.zeros((len(num_samples),), dtype=torch.float, device=tensors.device)
    for idx, (tensor, num_sample) in enumerate(zip(tensors, num_samples)):
        if num_sample <= 1:
            losses[idx] = 0.0
        else:
            loss = pairwise_cosine_similarity(tensor[:num_sample,:]).mean()
            losses[idx] = loss

    return _reduction(losses, reduction)