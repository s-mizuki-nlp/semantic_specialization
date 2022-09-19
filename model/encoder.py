#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os.path
import warnings
from typing import Optional, Dict, Any, Union, Tuple, List, Iterable

import io, math, copy
import numpy as np
import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, n_dim_in: int,
                 n_dim_out: Optional[int] = None,
                 n_dim_hidden: Optional[int] = None,
                 n_layer: Optional[int] = 2,
                 activation_function = torch.relu,
                 bias: bool = False,
                 init_zeroes: bool = False,
                 distinguish_gloss_context_embeddings: bool = False,
                 **kwargs):
        """
        multi-layer dense neural network with artibrary activation function
        output = Dense(iter(Activation(Dense())))(input)

        :param n_dim_in: input dimension size
        :param n_dim_out: output dimension size. DEFAULT: n_dim_in
        :param n_dim_hidden: hidden layer dimension size. DEFAULT: n_dim_in
        :param n_layer: number of layers. DEFAULT: 2
        :param activation_function: activation function. e.g. torch.relu
        """
        super().__init__()

        n_dim_out = n_dim_in if n_dim_out is None else n_dim_out
        n_dim_hidden = n_dim_in if n_dim_hidden is None else n_dim_hidden

        self._n_dim_in = n_dim_in
        self._n_hidden = n_layer
        self._n_dim_out = n_dim_out
        self._bias = bias
        self._distinguish_gloss_context_embeddings = distinguish_gloss_context_embeddings
        self._lst_dense = []
        if distinguish_gloss_context_embeddings:
            embedding_dim = kwargs.get("embedding_dim", None)
            embedding_dim = int(n_dim_in//8) if embedding_dim is None else embedding_dim
            self._embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        else:
            self._embedding = None
            embedding_dim = 0

        for k in range(n_layer):
            n_in = n_dim_in+embedding_dim if k==0 else n_dim_hidden
            n_out = n_dim_out if k==(n_layer - 1) else n_dim_hidden
            dense_layer = nn.Linear(n_in, n_out, bias=bias)
            if init_zeroes:
                nn.init.uniform_(dense_layer.weight, -1/math.sqrt(n_in*1000), 1/math.sqrt(n_in*1000))
            self._lst_dense.append(dense_layer)
        self._activation = activation_function
        self._layers = nn.ModuleList(self._lst_dense)

    def forward(self, x: torch.Tensor, is_gloss_embeddings: bool = None):

        if self._distinguish_gloss_context_embeddings:
            # concat gloss_or_context embedding.
            assert is_gloss_embeddings is not None, f"argument `is_gloss_embeddings` must be specified because `distinguish_gloss_context_embeddings=True`"
            idx = int(is_gloss_embeddings)
            # shape = (n_batch, (n_sample), 1)
            shape = x.shape[:-1] + (1,)
            t_e = self._embedding(torch.tensor(idx).to(x.device))
            t_e = torch.tile(t_e, shape)
            x = torch.cat([x, t_e], dim=-1)

        for k, dense in enumerate(self._layers):
            if k == 0:
                h = self._activation(dense(x))
            elif k == (self._n_hidden-1):
                h = dense(h)
            else:
                h = self._activation(dense(h))

        return h

    def predict(self, x, is_gloss_embeddings: bool = None):
        with torch.no_grad():
            return self.forward(x, is_gloss_embeddings)

    def verbose(self):
        return {}


class NormRestrictedShift(nn.Module):

    def __init__(self, n_dim_in: int,
                 n_layer: Optional[int] = 2,
                 n_dim_hidden: Optional[int] = None,
                 activation_function = torch.relu,
                 max_l2_norm_value: Optional[float] = None,
                 max_l2_norm_ratio: Optional[float] = None,
                 init_zeroes: bool = False,
                 bias: bool = False,
                 distinguish_gloss_context_embeddings: bool = False,
                 **kwargs):
        """
        this module shifts input vector up to max L2 norm.
        let x as input, d as number of dimensions, \sigma as sigmoid function, F(x) as the multi-layer perceptron, output will be written as follows.
        output = x + \epsilon (2 \sigma(F(x)) - 1); \epsilon = max_l2_norm / \sqr{d}

        :param n_dim_in: input dimension size
        :param max_l2_norm_value: maximum absolute L2 norm value of shift vector.
        :param max_l2_norm_ratio: maximum relative L2 norm value of shift vector relative to input vector.
        :param n_dim_hidden: MLP hidden layer dimension size
        :param n_layer: MLP number of layers
        :param activation_function: MLP activation function. e.g. torch.relu
        :param init_zeroes: initialize MLP with very small values. i.e., initial shift will be almost zero.
        """
        super().__init__()

        assert (max_l2_norm_value is not None) or (max_l2_norm_ratio is not None), f"either `max_l2_norm_value` or `max_l2_norm_ratio` must be specified."
        assert (max_l2_norm_value is None) or (max_l2_norm_ratio is None), f"you can't specify both `max_l2_norm_value` and `max_l2_norm_ratio` simultaneously."

        self._ffn = MultiLayerPerceptron(n_dim_in=n_dim_in, n_dim_out=n_dim_in, n_dim_hidden=n_dim_hidden, n_layer=n_layer,
                                         activation_function=activation_function, bias=bias, init_zeroes=init_zeroes,
                                         distinguish_gloss_context_embeddings=distinguish_gloss_context_embeddings)
        self._max_l2_norm_value = max_l2_norm_value
        self._max_l2_norm_ratio = max_l2_norm_ratio

    def forward(self, x, is_gloss_embeddings: bool = None):
        z = self._ffn.forward(x, is_gloss_embeddings)

        # limit l2-norm up to 1.0
        # z_norm = ||z||
        z_norm = torch.linalg.norm(z, ord=2, dim=-1, keepdim=True)
        # z_denom = 1.0 if ||z|| < 1.0 else ||z||
        z_denom = torch.maximum(torch.ones_like(z_norm, dtype=torch.float), z_norm)

        if self._max_l2_norm_value is not None:
            # epsilon: (1,)
            epsilon = self._max_l2_norm_value
        elif self._max_l2_norm_ratio is not None:
            # epsilon: (n,*,1)
            epsilon = self._max_l2_norm_ratio * torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)

        # dx = \epsilon * ||x|| * z / max(1.0, ||z||)
        dx = epsilon * z / z_denom
        y = x + dx

        return y

    def predict(self, x, is_gloss_embeddings: bool = None):
        with torch.no_grad():
            return self.forward(x, is_gloss_embeddings)


class Identity(nn.Module):

    def __init__(self, assign_dummy_parameter: bool = False, **kwargs):
        """
        multi-layer dense neural network with artibrary activation function
        output = Dense(iter(Activation(Dense())))(input)

        :param n_dim_in: input dimension size
        :param n_dim_out: output dimension size
        :param n_dim_hidden: hidden layer dimension size
        :param n_layer: number of layers
        :param activation_function: activation function. e.g. torch.relu
        """
        super().__init__()
        if assign_dummy_parameter:
            self.dummy = nn.Parameter(torch.zeros((1,)), requires_grad=False)

    def forward(self, x, **kwargs):
        return x

    def predict(self, x, **kwargs):
        with torch.no_grad():
            return self.forward(x)