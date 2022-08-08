#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Callable, Iterable, Any, List, Union, Set, Tuple
from pprint import pprint

import warnings
from collections import defaultdict
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

from dataset import WordInContextTaskDataset
from model.encoder import NormRestrictedShift


class WordInContextTaskEvaluatorBase(object, metaclass=ABCMeta):

    def __init__(self,
                 development_dataset: WordInContextTaskDataset,
                 evaluation_dataset: WordInContextTaskDataset,
                 verbose: bool = False):

        # development dataset -> used for fine-tuning classification threshold
        self._development_dataset = development_dataset
        self._evaluation_dataset = evaluation_dataset

        self.verbose = verbose

    def calc_optimal_threshold_accuracy(self, y_true, similarity_value, verbose: bool = True, **kwargs):

        # compute the threshold that maximizes accuracy using receiver operating curve.
        v_fpr, v_tpr, v_threshold = roc_curve(y_true=y_true, y_score=similarity_value, **kwargs)
        n_sample = len(y_true)
        n_positive = np.sum(np.array(y_true) == True)
        n_negative = n_sample - n_positive
        v_accuracy = (v_tpr*n_positive + (1-v_fpr)*n_negative)/n_sample

        idx = np.nanargmax(v_accuracy)
        threshold_opt = v_threshold[idx]

        if verbose:
            report = {
                "threshold_opt": threshold_opt,
                "tpr": v_tpr[idx],
                "fpr": v_fpr[idx],
                "accuracy": v_accuracy[idx]
            }
            pprint(report)

        return threshold_opt

    def grid_search_optimal_threshold_accuracy(self, y_true, similarity_value,
                                               sim_min=-1.0, sim_max=1.0, step=0.02,
                                               verbose: bool = False, **kwargs):
        # grid-search using optimal threshold
        v_thresholds = np.arange(start=sim_min, stop=sim_max+1E-15, step=step)

        dict_accuracy = {}
        for threshold in v_thresholds:
            v_prediction = similarity_value > threshold
            acc = accuracy_score(y_true=y_true, y_pred=v_prediction)
            dict_accuracy[threshold] = acc

        # sort by accuracy descending order
        threshold_opt = sorted(dict_accuracy, key=dict_accuracy.get, reverse=True)[0]

        if verbose:
            report = {
                "threshold_opt": threshold_opt,
                "accuracy": dict_accuracy[threshold_opt]
            }
            print("=== tureshold tuning ===")
            pprint(report)

        return threshold_opt

    @abstractmethod
    def calc_similarity(self, input: Dict[str, Any], **kwargs) -> float:
        pass

    def compute_metrics(self, ground_truthes: Iterable[bool], predictions: Iterable[bool], similarities: Optional[Iterable[float]] = None) -> Dict[str, float]:
        dict_results = {
            "accuracy": accuracy_score(y_true=ground_truthes, y_pred=predictions),
            "auc": roc_auc_score(y_true=ground_truthes, y_score=similarities) if similarities is not None else None
        }
        return dict_results

    def assertion(self):
        return True

    def _get_attr_key_and_values(self, set_attr_names: Set[str], example: Dict[str, str], concat="|"):
        attr_keys = concat.join([attr_name for attr_name in set_attr_names])
        attr_values = concat.join([example[attr_name] for attr_name in set_attr_names])
        return attr_keys, attr_values

    def batch_calc_similarity(self, dataset: WordInContextTaskDataset, return_ground_truth: bool = False, **kwargs) -> Union[List[float], Tuple[List[float], List[bool]]]:
        lst_similarities = []
        lst_ground_truthes = []
        for record in dataset:
            similarity = self.calc_similarity(input=record,  **kwargs)
            ground_truth = record["ground_truth_label"] if dataset.has_ground_truth else None

            lst_similarities.append(similarity)
            lst_ground_truthes.append(ground_truth)

        if return_ground_truth:
            return lst_similarities, lst_ground_truthes
        else:
            return lst_similarities

    def __len__(self):
        return len(self._evaluation_dataset)

    def evaluate(self, **kwargs) -> [Dict[str, float], List[float], List[bool], Optional[List[bool]]]:
        assert self.assertion(), f"assertion failed."

        # optimize similarity threshold
        lst_similarities, lst_ground_truthes = self.batch_calc_similarity(dataset=self._development_dataset, return_ground_truth=True, **kwargs)
        if "step" in kwargs:
            step = kwargs.pop("step")
        else:
            step = 0.02
        threshold_opt = self.grid_search_optimal_threshold_accuracy(y_true=lst_ground_truthes, similarity_value=lst_similarities,
                                                                    sim_min=-1.0, sim_max=1.0, step=step, verbose=self.verbose,
                                                                    **kwargs)

        # compute similarities and do prediction using optimal threshold
        if self._evaluation_dataset.has_ground_truth:
            lst_similarities, lst_ground_truthes = self.batch_calc_similarity(dataset=self._evaluation_dataset, return_ground_truth=True)
        else:
            warnings.warn(f"evalset doesn't have ground truth. we just return prediction results.")
            lst_similarities = self.batch_calc_similarity(dataset=self._evaluation_dataset, return_ground_truth=False)

        lst_predictions = (np.array(lst_similarities) > threshold_opt).tolist()
        if self._evaluation_dataset.has_ground_truth:
            dict_summary = self.compute_metrics(predictions=lst_predictions, ground_truthes=lst_ground_truthes, similarities=lst_similarities)
        else:
            dict_summary = {}
        dict_summary["threshold_opt"] = threshold_opt

        if self._evaluation_dataset.has_ground_truth:
            return dict_summary, lst_similarities, lst_predictions, lst_ground_truthes
        else:
            return dict_summary, lst_similarities, lst_predictions, None


class WiCTaskByEmbeddingSimilarityEvaluator(WordInContextTaskEvaluatorBase):

    def __init__(self,
                 development_dataset: WordInContextTaskDataset,
                 evaluation_dataset: WordInContextTaskDataset,
                 projection_head: Optional[NormRestrictedShift] = None,
                 context_embedding_entity: str = "word",
                 is_gloss_embeddings: bool = False,
                 verbose: bool = False):
        """
        Evaluate Word-in-Context task using word / sentence similarity

        Args:
            development_dataset:
            evaluation_dataset:
            context_embedding_entity: "sentence" or "word". "sentence" uses average of all subwords in a sentence while "word" uses subwords in a word of interest.
            verbose:
        """

        super().__init__(development_dataset=development_dataset, evaluation_dataset=evaluation_dataset, verbose=verbose)
        available_context_embedding_entity = ("word", "sentence")
        assert context_embedding_entity in available_context_embedding_entity, f"invalid `context_embedding_entity` was specified. valid values are: {available_context_embedding_entity}"
        self._context_embedding_entity = context_embedding_entity
        self._projection_head = projection_head
        self._is_gloss_embeddings = is_gloss_embeddings
        if projection_head is not None:
            self._device = next(projection_head.parameters()).device
        else:
            self._device = "cpu"

    def calc_similarity(self, input: Dict[str, Any], **kwargs) -> float:
        if self._context_embedding_entity == "word":
            t_x = input["entity_span_avg_vector_x"].to(self._device)
            t_y = input["entity_span_avg_vector_y"].to(self._device)
        elif self._context_embedding_entity == "sentence":
            t_x = input["sentence_vector_x"].to(self._device)
            t_y = input["sentence_vector_y"].to(self._device)

        if self._projection_head is not None:
            t_x = self._projection_head.predict(t_x, is_gloss_embeddings=self._is_gloss_embeddings)
            t_y = self._projection_head.predict(t_y, is_gloss_embeddings=self._is_gloss_embeddings)

        sim = F.cosine_similarity(t_x, t_y, dim=-1).item()

        return sim