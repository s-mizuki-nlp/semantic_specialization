#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn


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