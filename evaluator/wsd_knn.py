#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Dict, Any, Iterable, Optional, Set, Union, Tuple, List

import numpy as np
import torch
from nltk.corpus import wordnet as wn

from .wsd_baseline import MostFrequentSenseWSDTaskEvaluator, numeric
from dataset import WSDTaskDataset
from dataset.gloss_sref import SREFLemmaEmbeddingsDataset
from dataset.utils import tensor_to_numpy, numpy_to_tensor, batch_tile
from model.similarity import CosineSimilarity, DotProductSimilarity


class FrozenBERTKNNWSDTaskEvaluator(MostFrequentSenseWSDTaskEvaluator):

    def __init__(self,

                 evaluation_dataset: WSDTaskDataset,
                 lemma_key_embeddings_dataset: SREFLemmaEmbeddingsDataset,
                 gloss_projection_head: torch.nn.Module,
                 context_projection_head: Optional[torch.nn.Module] = None,
                 target_pos: Tuple[str] = ("n","v","a","s","r"),
                 similarity_module: Union[str, torch.nn.Module] = "cosine",
                 evaluation_category: str = "lemma",
                 ground_truth_lemma_keys_field_name: str = "ground_truth_lemma_keys",
                 entity_embedding_field_name: str = "entity_span_avg_vectors",
                 breakdown_attributes: Optional[Iterable[Set[str]]] = None,
                 device: Optional[str] = "cpu",
                 verbose: bool = False,
                 **kwargs_dataloader):
        super().__init__(
            evaluation_dataset=evaluation_dataset,
            evaluation_category=evaluation_category,
            ground_truth_lemma_keys_field_name=ground_truth_lemma_keys_field_name,
            breakdown_attributes=breakdown_attributes,
            device=device,
            verbose=verbose,
            **kwargs_dataloader)

        self._entity_embedding_field_name = entity_embedding_field_name
        self._lemma_key_embeddings_dataset = lemma_key_embeddings_dataset
        self._target_pos = target_pos
        self._gloss_prjection_head = gloss_projection_head
        if context_projection_head is None:
            warnings.warn(f"gloss_projection_head will be used as context_projection_head.")
            self._context_projection_head = gloss_projection_head
        else:
            self._context_projection_head = context_projection_head

        if isinstance(similarity_module, str):
            if similarity_module == "cosine":
                self._similarity_module = CosineSimilarity()
            elif similarity_module == "dot":
                self._similarity_module = DotProductSimilarity()
            else:
                raise ValueError(f"unknown `similarity_module` name: {similarity_module}")
        else:
            self._similarity_module = similarity_module

    def get_lemma_key_embedding(self, lemma_key: str) -> np.ndarray:
        """
        get lemma key embedding which is precomputed using WordNet gloss corpus.

        Args:
            lemma_key: lemma key.

        Returns: lemma key embedding. shape: (n_dim, )

        """
        lst_records = self._lemma_key_embeddings_dataset.get_records_by_lemma_key(lemma_key=lemma_key)
        assert len(lst_records) == 1, f"record must be unique for each lemma key."
        record = lst_records[0]
        assert lemma_key in record["ground_truth_lemma_keys"], f"lemma key lookup failure: {lemma_key}"

        return record["embeddings"]

    def get_lemma_key_embeddings(self, lst_lemma_keys: List[str]) -> torch.Tensor:
        it = map(self.get_lemma_key_embedding, lst_lemma_keys)
        v_embs = np.stack(list(it))
        return numpy_to_tensor(v_embs)

    def predict(self, input: Dict[str, Any],
                mfs_reorder_by_lemma_counts: bool = False,
                output_tie_lemma: bool = False) -> Iterable[str]:
        """
        predict the most plausible sense based on conditional probability

        @param input:
        @param use_generated_code_probability:
        @param apply_one_hot_encoding: convert from continuous relaxed repr. to one-hot repr.
        @return:
        """

        # when ties happen, then we fall back to most frequent sense.
        ties_fallback_to_mfs = (output_tie_lemma == False)

        lemma_name = input["lemma"]
        pos = input["pos"]

        # if unsupported part-of-speech tag, then fall back to most frequent sense method.
        if pos not in self._target_pos:
            return super().predict(input, reorder_by_lemma_counts=mfs_reorder_by_lemma_counts, output_tie_lemma=output_tie_lemma)

        # get candidates
        lst_candidate_lemmas = self.get_candidate_lemmas_from_wordnet(lemma_name, pos)
        n_candidates = len(lst_candidate_lemmas)
        lst_candidate_lemma_keys = [lemma.key() for lemma in lst_candidate_lemmas]

        # lookup candidate lemma key embeddings from WordNet gloss embeddings dataset
        device = input["entity_embeddings"].device
        t_candidate_lemma_key_embeddings = self.get_lemma_key_embeddings(lst_candidate_lemma_keys).to(device)

        # query embedding:
        # shape: (1, n_dim) if entity_embedding_field_name = "entity_span_avg_vectors"
        # shape: (1, n_subwords, n_dim) if entity_embedding_field_name = "entity_span_avg_vectors"
        t_query_embedding = input[self._entity_embedding_field_name]
        if t_query_embedding.ndim == 3:
            # average along subword dimension (=2nd dimension)
            t_query_embedding = torch.mean(t_query_embedding, dim=1)

        # inference using k-NN method.
        t_candidate_lemma_key_embeddings = self._gloss_prjection_head.predict(t_candidate_lemma_key_embeddings)
        t_query_embedding = self._context_projection_head.predict(t_query_embedding)
        t_query_embeddings = torch.tile(t_query_embedding, (n_candidates, 1))
        assert t_candidate_lemma_key_embeddings.shape == t_query_embeddings.shape, f"query and candidate shape mismatch."

        # calculate similarity
        t_sim_score = self._similarity_module(t_query_embeddings, t_candidate_lemma_key_embeddings)
        lst_metric_scores = tensor_to_numpy(t_sim_score).tolist()

        # return top-k lemma keys
        if ties_fallback_to_mfs:
            lst_sense_freq_scores = self.score_by_sense_frequency(lst_lemmas=lst_candidate_lemmas, reorder_by_lemma_count=mfs_reorder_by_lemma_counts)
            # order by metric first, then order by wordnet frequency rank.
            lst_scores = (lst_metric_scores, lst_sense_freq_scores)
        else:
            lst_scores = lst_metric_scores
        return self.return_top_k_lemma_keys(lst_candidate_lemmas, lst_scores, multiple_output=output_tie_lemma)

    def __iter__(self):
        it = super().__iter__()
        for inputs_for_predictor, inputs_for_evaluator, ground_truthes, predictions, dict_metrics in it:
            if self.verbose:
                print(f"ground truth: {inputs_for_evaluator['lemma']}|{inputs_for_predictor['pos']}")
                lexnames = list(map(self._lemma_to_lexname, ground_truthes))
                ground_truth_synset_ids = list(map(self._lemma_to_synset_id, ground_truthes))
                for lexname, synset_id, ground_truth in zip(lexnames, ground_truth_synset_ids, ground_truthes):
                    print(f"\t{lexname}-{synset_id}-{ground_truth}")
                print("-------------")
            yield inputs_for_predictor, inputs_for_evaluator, ground_truthes, predictions, dict_metrics

    def _print_verbose(self, lst_tup_lemma_and_scores: List[Tuple[wn.lemma, Union[numeric, Tuple[numeric]]]]):
        print(f"similarity_module: {self._similarity_module}")
        print(f"candidates:")
        for lemma, scores in lst_tup_lemma_and_scores:
            lexname = lemma.synset().lexname()
            synset_id = lemma.synset().name()
            if isinstance(scores, float):
                print(f"\t{lexname}-{synset_id}-{lemma.key()}: {scores:1.6f}")
            else:
                print(f"\t{lexname}-{synset_id}-{lemma.key()}: {scores}")
