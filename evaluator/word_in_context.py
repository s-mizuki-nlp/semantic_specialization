#!/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Callable, Iterable, Any, List, Union, Set, Tuple
from pprint import pprint

import warnings
from collections import defaultdict
from nltk.corpus import wordnet as wn
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

from dataset import WordInContextTaskDataset
from dataset.gloss_embeddings import SREFLemmaEmbeddingsDataset
from dataset.utils import tensor_to_numpy, numpy_to_tensor
from dataset_preprocessor.utils_wordnet_gloss import wu_palmer_similarity_lemma_key_pair
from model.encoder import NormRestrictedShift
from model.similarity import CosineSimilarity
from .wsd_heuristics import TryAgainMechanism, TryAgainMechanismWithCoarseSenseInventory


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
            print("=== threshold tuning ===")
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

        ## if explicitly specified, use as-is
        threshold_opt = kwargs.get("threshold_opt", None)
        ## otherwise, we estimate by grid-search using development set.
        if threshold_opt is None:
            step = kwargs.pop("step") if "step" in kwargs else 0.02
            lst_similarities, lst_ground_truthes = self.batch_calc_similarity(dataset=self._development_dataset, return_ground_truth=True, **kwargs)
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


class WiCTaskByNNSenseSimilarityEvaluator(WordInContextTaskEvaluatorBase):

    def __init__(self,
                 development_dataset: WordInContextTaskDataset,
                 evaluation_dataset: WordInContextTaskDataset,
                 lemma_key_embeddings_dataset: SREFLemmaEmbeddingsDataset,
                 sense_similarity_metric: str,
                 try_again_mechanism: bool = False,
                 gloss_projection_head: Optional[torch.nn.Module] = None,
                 context_projection_head: Optional[torch.nn.Module] = None,
                 kwargs_try_again_mechanism: Optional[Dict] = None,
                 device: Optional[str] = "cpu",
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

        self._lemma_key_embeddings_dataset = lemma_key_embeddings_dataset
        self._gloss_projection_head = gloss_projection_head
        self._context_projection_head = context_projection_head
        self._sense_similarity_metric = sense_similarity_metric

        available_sense_similarity_metric = ("identity", "wu_palmer")
        assert sense_similarity_metric in available_sense_similarity_metric, f"invalid `sense_similarity_metric` is specified. available values are: {available_sense_similarity_metric}"

        if lemma_key_embeddings_dataset.is_projected:
            warnings.warn(f"we ignore gloss_projection_head because lemma key embeddings are already projected.")
            self._apply_gloss_projection = False
            self._gloss_projection_head = None
        else:
            if (gloss_projection_head is not None) and (gloss_projection_head.__class__.__name__ != "Identity"):
                print("apply gloss projection head to gloss embeddings...")
                self._lemma_key_embeddings_dataset.project_gloss_embeddings(gloss_projection_head=gloss_projection_head, chunksize=1024)
                self._apply_gloss_projection = False
                self._gloss_projection_head = None
            else:
                warnings.warn(f"no gloss projection is applied.")
                self._apply_gloss_projection = False

        self._similarity_module = CosineSimilarity(temperature=1.0)

        # try-again mechanism
        if try_again_mechanism is None:
            self._try_again_mechanism = None
        elif isinstance(try_again_mechanism, (TryAgainMechanism, TryAgainMechanismWithCoarseSenseInventory)):
            self._try_again_mechanism = try_again_mechanism
        elif isinstance(try_again_mechanism, bool):
            if try_again_mechanism:
                if isinstance(kwargs_try_again_mechanism, dict):
                    _cfg = kwargs_try_again_mechanism
                else:
                    _cfg = {
                        "exclude_common_semantically_related_synsets": True,
                        "lookup_first_lemma_sense_only": False,
                        "average_similarity_in_synset": False,
                        "exclude_oneselves_for_noun_and_verb": True,
                        "do_not_fix_synset_degeneration_bug": False,
                        "semantic_relation": "all-relations"
                    }
                self._try_again_mechanism = TryAgainMechanism(lemma_key_embeddings_dataset=lemma_key_embeddings_dataset,
                                                              similarity_metric="cosine",
                                                              device=device,
                                                              verbose=verbose,
                                                              **_cfg)
            else:
                self._try_again_mechanism = None
        else:
            raise ValueError(f"invalid `try_again_mechanism` argument: {type(try_again_mechanism)}")

    def get_candidate_lemmas_from_wordnet(self, str_lemma: str, pos: str) -> List[wn.lemma]:
        lst_lemmas = wn.lemmas(str_lemma, pos=pos)
        assert len(lst_lemmas) > 0, f"unknown lemma: {str_lemma}|{pos}"
        return lst_lemmas

    def get_lemma_key_embedding(self, lemma_key: str) -> np.ndarray:
        """
        get lemma key embedding which is precomputed using WordNet gloss corpus.

        Args:
            lemma_key: lemma key.

        Returns: lemma key embedding. shape: (n_dim, )

        """
        lst_records = self._lemma_key_embeddings_dataset.get_records_by_lemma_key(lemma_key=lemma_key)
        assert len(lst_records) == 1, f"record must be unique for each lemma key: {lemma_key}"
        record = lst_records[0]
        assert lemma_key in record["ground_truth_lemma_keys"], f"lemma key lookup failure: {lemma_key}"

        return record["embeddings"]

    def get_lemma_key_embeddings(self, lst_lemma_keys: List[str]) -> torch.Tensor:
        it = map(self.get_lemma_key_embedding, lst_lemma_keys)
        v_embs = np.stack(list(it))
        return numpy_to_tensor(v_embs)

    def return_top_k_lemma_keys(self, lst_lemmas: List[wn.lemma], lst_scores: Union[List[float], Tuple[List[float]]],
                                multiple_output: bool) -> List[str]:
        if isinstance(lst_scores, tuple):
            lst_scores = list(zip(*lst_scores))
        lst_tup_lemma_and_scores = list(zip(lst_lemmas, lst_scores))
        lst_tup_lemma_and_scores = sorted(lst_tup_lemma_and_scores, key=lambda tup: tup[1], reverse=True)

        if multiple_output:
            lst_keys = []; prev_scores = None
            for lemma, scores in lst_tup_lemma_and_scores:
                if (prev_scores is not None) and (scores < prev_scores):
                    break
                lst_keys.append(lemma.key())
                prev_scores = scores
            return lst_keys
        else:
            lemma, scores = lst_tup_lemma_and_scores[0]
            return [lemma.key()]

    def predict_sense(self, lemma_name: str, pos: str, context_embedding: torch.Tensor):
        """
        predict the most similar sense based on gloss-context similarity

        @return:
        """

        # get candidates
        lst_candidate_lemmas = self.get_candidate_lemmas_from_wordnet(lemma_name, pos)
        n_candidates = len(lst_candidate_lemmas)
        lst_candidate_lemma_keys = [lemma.key() for lemma in lst_candidate_lemmas]

        # lookup candidate lemma key embeddings from WordNet gloss embeddings dataset
        device = context_embedding.device
        t_candidate_lemma_key_embeddings = self.get_lemma_key_embeddings(lst_candidate_lemma_keys).to(device)

        # query embedding:
        # shape: (1, n_dim) if entity_embedding_field_name = "entity_span_avg_vectors"
        # shape: (1, n_subwords, n_dim) if entity_embedding_field_name = "entities"
        t_query_embedding = context_embedding
        if t_query_embedding.ndim == 3:
            # average along subword dimension (=2nd dimension)
            t_query_embedding = torch.mean(t_query_embedding, dim=1)

        # project query(=context) embeddings if context_projection_head is specified.
        if self._context_projection_head is not None:
            t_query_embedding = self._context_projection_head.predict(t_query_embedding, is_gloss_embeddings=False)

        # inference using k-NN method.
        t_query_embeddings = torch.tile(t_query_embedding, (n_candidates, 1))
        assert t_candidate_lemma_key_embeddings.shape == t_query_embeddings.shape, f"query and candidate shape mismatch."

        # calculate similarity
        t_sim_score = self._similarity_module(t_query_embeddings, t_candidate_lemma_key_embeddings)
        lst_scores = tensor_to_numpy(t_sim_score).tolist()

        if self._try_again_mechanism is not None:
            if t_query_embedding.ndim == 1:
                vec_query_embedding = tensor_to_numpy(t_query_embedding.unsqueeze(0))
            else:
                vec_query_embedding = tensor_to_numpy(t_query_embedding)
            _lst_candidate_lemma_keys, lst_scores = self._try_again_mechanism.try_again_mechanism(
                # vec_query_embedding: (n_dim,)
                vec_query_embedding=vec_query_embedding,
                pos=pos,
                lst_candidate_lemma_keys=lst_candidate_lemma_keys,
                lst_candidate_similarities=lst_scores,
                top_k_candidates=2)

        lst_predictions = self.return_top_k_lemma_keys(lst_candidate_lemmas, lst_scores, multiple_output=False)
        return lst_predictions[0]

    def calc_similarity(self, input: Dict[str, Any], **kwargs) -> float:
        lemma_x = self.predict_sense(lemma_name=input["lemma"], pos=input["pos"], context_embedding=input["entity_span_avg_vector_x"])
        lemma_y = self.predict_sense(lemma_name=input["lemma"], pos=input["pos"], context_embedding=input["entity_span_avg_vector_y"])

        if self._sense_similarity_metric == "identity":
            sim = 1.0 if lemma_x == lemma_y else 0.0
        elif self._sense_similarity_metric == "wu_palmer":
            sim = wu_palmer_similarity_lemma_key_pair(lemma_key_x=lemma_x, lemma_key_y=lemma_y)
            if sim is None:
                sim = 0.0

        return sim

    def evaluate(self, **kwargs) -> [Dict[str, float], List[float], List[bool], Optional[List[bool]]]:
        if self._sense_similarity_metric == "identity":
            # we can deterministically specify optimal threshold as 0.5 because similarity is zero-one.
            return super().evaluate(threshold_opt=0.5)
        elif self._sense_similarity_metric == "wu_palmer":
            return super().evaluate()