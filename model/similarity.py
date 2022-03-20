#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math

import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class CosineSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self._temperature = temperature
        self._cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self._cosine(x, y) / self._temperature


class DotProductSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self._temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.sum(x*y, dim=-1) / self._temperature


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            s: coefficient
            m: margin [Rad]
            return: s*cos(theta + m) for positive example, s*cos(theta) for negative example.
        """
    def __init__(self, s=10.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        # cosine additive theorem: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_hard_examples: bool):
        cos_theta = cosine_similarity(x, y, dim=-1)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(0, 1))
        # phi = cos(theta + m)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        if is_hard_examples:
            # return s*cos(theta + m)
            output = phi * self.s
        else:
            # return s*cos(theta)
            output = cos_theta * self.s

        return output