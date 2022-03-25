#!/usr/bin/env python
# -*- coding:utf-8 -*-

import unittest

import numpy as np
import torch
from dataset.utils import tensor_to_numpy
from model.similarity import CosineSimilarity, DotProductSimilarity, ArcMarginProduct


class SimilarityModuleTestCase(unittest.TestCase):

    def setUp(self) -> None:
        n = 4
        n_dim = 6

        t_x = torch.rand((n, n_dim))
        t_y = torch.rand((n, n_dim))

        mat_x = tensor_to_numpy(t_x)
        mat_y = tensor_to_numpy(t_y)

        self._t_x = t_x
        self._t_y = t_y
        self._mat_x = mat_x
        self._mat_y = mat_y

    def _compare(self, expected, actual, rtol=1E-2, atol=1E-4):
        return np.allclose(expected, actual, rtol=rtol, atol=atol)

    def test_cosine_similarity(self):
        temperature = 2.0
        t_x, t_y = self._t_x, self._t_y
        mat_x, mat_y = self._mat_x, self._mat_y
        cosine_similarity = CosineSimilarity(temperature = temperature)

        expected = (mat_x * mat_y).sum(axis=-1) / (np.linalg.norm(mat_x, axis=-1) * np.linalg.norm(mat_y, axis=-1))
        expected /= temperature

        actual = cosine_similarity(t_x, t_y)
        actual = tensor_to_numpy(actual)

        self.assertTrue(self._compare(expected, actual))

    def test_dot_similarity(self):
        temperature = 5.0
        t_x, t_y = self._t_x, self._t_y
        mat_x, mat_y = self._mat_x, self._mat_y
        dot_similarity = DotProductSimilarity(temperature = temperature)

        expected = (mat_x * mat_y).sum(axis=-1) / temperature

        actual = dot_similarity(t_x, t_y)
        actual = tensor_to_numpy(actual)

        self.assertTrue(self._compare(expected, actual))

    def _arc_margin_similarity(self, is_hard_examples: bool):
        temperature = 0.1
        margin = 0.5
        t_x, t_y = self._t_x, self._t_y
        mat_x, mat_y = self._mat_x, self._mat_y
        arc_similarity = ArcMarginProduct(margin=margin, temperature=temperature)

        cos_theta = (mat_x * mat_y).sum(axis=-1) / (np.linalg.norm(mat_x, axis=-1) * np.linalg.norm(mat_y, axis=-1))
        theta = np.arccos(cos_theta)
        if is_hard_examples:
            expected = np.cos(theta + margin) / temperature
        else:
            expected = cos_theta / temperature

        actual = arc_similarity(t_x, t_y, is_hard_examples=is_hard_examples)
        actual = tensor_to_numpy(actual)

        return self._compare(expected, actual)

    def test_arc_margin_similarity(self):

        for is_hard_examples in (True, False):
            with self.subTest(msg=f"is_hard_examples: {is_hard_examples}"):
                self.assertTrue(self._arc_margin_similarity(is_hard_examples))


if __name__ == '__main__':
    unittest.main()
