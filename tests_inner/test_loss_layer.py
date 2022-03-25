#!/usr/bin/env python
# -*- coding:utf-8 -*-

import unittest
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.utils import tensor_to_numpy
from model.loss import ContrastiveLoss
from model.similarity import CosineSimilarity

class ContrastiveLossTestCase(unittest.TestCase):

    def setUp(self) -> None:
        n = 4
        n_dim = 6
        lst_n_neg = [1,2,7,3] # number of negative examples for each entry(=n).

        t_n_neg = torch.tensor(lst_n_neg)

        t_targets = torch.rand((n, n_dim))
        t_positives = torch.rand((n, n_dim))
        lst_negatives = [torch.rand(n_neg, n_dim) for n_neg in lst_n_neg]
        t_negatives = pad_sequence(lst_negatives, batch_first=True, padding_value=0.0)

        vec_n_neg = tensor_to_numpy(t_n_neg)
        mat_targets = tensor_to_numpy(t_targets)
        mat_positives = tensor_to_numpy(t_positives)
        arr_negatives = tensor_to_numpy(t_negatives)

        for var_name in "t_targets,t_positives,t_negatives,t_n_neg".split(","):
            self.__setattr__("_"+var_name, locals()[var_name])

        for var_name in "mat_targets,mat_positives,arr_negatives,vec_n_neg".split(","):
            self.__setattr__("_"+var_name, locals()[var_name])

    def _compare(self, expected, actual, rtol=1E-2, atol=1E-4):
        return np.allclose(expected, actual, rtol=rtol, atol=atol)

    def test_loss_with_cosine_similarity(self):

        t_targets, t_positives, t_negatives = self._t_targets, self._t_positives, self._t_negatives
        t_n_neg = self._t_n_neg

        mat_targets, mat_positives, arr_negatives = self._mat_targets, self._mat_positives, self._arr_negatives
        vec_n_neg = self._vec_n_neg

        similarity = CosineSimilarity(temperature=1.0)
        loss = ContrastiveLoss(similarity_module=similarity, reduction="none")

        def _cosine(mat_x, mat_y):
            cos_theta = (mat_x * mat_y).sum(axis=-1) / (np.linalg.norm(mat_x, axis=-1) * np.linalg.norm(mat_y, axis=-1))
            return cos_theta

        vec_sim_pos = _cosine(mat_targets, mat_positives)
        expected = []
        for vec_target, mat_negatives, num_neg, sim_pos in zip(mat_targets, arr_negatives, vec_n_neg, vec_sim_pos):
            _targets = np.tile(vec_target, (num_neg, 1))
            vec_sim_neg = _cosine(_targets, mat_negatives[:num_neg,:])
            vec_logits = np.append(vec_sim_neg, sim_pos)
            neg_llk = - (sim_pos - np.log(np.sum(np.exp(vec_logits))))
            expected.append(neg_llk)
        expected = np.array(expected)

        actual = loss(queries=t_targets, positives=t_positives, negatives=t_negatives, num_negative_samples=t_n_neg)
        actual = tensor_to_numpy(actual)

        self.assertTrue(self._compare(expected, actual))

if __name__ == '__main__':
    unittest.main()
