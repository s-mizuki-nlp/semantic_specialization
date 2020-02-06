#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union
from collections import defaultdict, Counter
from functools import lru_cache
import warnings
import networkx as nx
import numpy as np
import progressbar
import random

from .lexical_knowledge import HyponymyDataset

class BasicTaxonomy(object):

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as a DAG
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset if record["distance"] == 1.0)
        self.build_directed_acyclic_graph(iter_hyponymy_pairs)
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset)
        self.record_ancestors_and_descendeants(iter_hyponymy_pairs)

    @property
    def dag(self):
        return self._dag

    @property
    def nodes(self):
        return self._nodes

    @property
    def trainset_ancestors(self):
        return self._trainset_ancestors

    @property
    def trainset_descendants(self):
        return self._trainset_descendants

    def build_directed_acyclic_graph(self, iter_hyponymy_pairs: Iterable[Tuple[str, str]]):
        """
        build taxonomy as a DAG based on the set of hyponymy relations

        @param iter_hyponymy_pairs: iterable of the tuple (hypernym, hyponym)
        """
        graph = nx.DiGraph()
        graph.add_edges_from(iter_hyponymy_pairs)

        self._dag = graph
        self._cache_root_nodes = {}
        self._nodes = set(graph.nodes)

    def record_ancestors_and_descendeants(self, iter_hyponymy_pairs):
        self._hyponym_frequency = Counter()
        self._hypernym_frequency = Counter()
        self._trainset_ancestors = defaultdict(set)
        self._trainset_descendants = defaultdict(set)
        for hypernym, hyponym in iter_hyponymy_pairs:
            self._trainset_ancestors[hyponym].add(hypernym)
            self._trainset_descendants[hypernym].add(hyponym)
            self._hyponym_frequency[hyponym] += 1
            self._hypernym_frequency[hypernym] += 1

    def _find_root_nodes(self, graph) -> Set[str]:
        hash_value = graph.__hash__() + graph.number_of_nodes()
        if hash_value in self._cache_root_nodes:
            return self._cache_root_nodes[hash_value]

        root_nodes = set([k for k,v in graph.in_degree() if v == 0])
        self._cache_root_nodes[hash_value] = root_nodes
        return root_nodes

    def remove_isolated_subgraphs(self, minimum_number_of_nodes: int = 3):
        for root_node in self._find_root_nodes(graph=self.dag):
            descendents = self.hyponyms_and_self(root_node)
            if len(descendents) < minimum_number_of_nodes:
                self.dag.remove_nodes_from(descendents)

    def hyponym_frequency(self, entity, not_exist: int = 0):
        if isinstance(entity, Iterable):
            return [self._hyponym_frequency.get(e, not_exist) for e in entity]
        else:
            return self._hyponym_frequency.get(entity, not_exist)

    def hypernym_frequency(self, entity, not_exist: int = 0):
        if isinstance(entity, Iterable):
            return [self._hypernym_frequency.get(e, not_exist) for e in entity]
        else:
            return self._hypernym_frequency.get(entity, not_exist)

    def hypernyms(self, entity):
        return nx.ancestors(self.dag, entity).union(self.trainset_ancestors.get(entity, set()))

    def hyponyms(self, entity):
        return nx.descendants(self.dag, entity).union(self.trainset_descendants.get(entity, set()))

    @lru_cache(maxsize=10000)
    def hypernyms_and_hyponyms_and_self(self, entity):
        return self.hyponyms(entity) | self.hypernyms(entity) | {entity}

    @lru_cache(maxsize=10000)
    def hyponyms_and_self(self, entity):
        return self.hyponyms(entity) | {entity}

    @lru_cache(maxsize=10000)
    def co_hyponyms(self, entity):
        graph = self.dag
        if entity not in graph:
            return {}
        direct_root_nodes = nx.ancestors(graph, entity) & self._find_root_nodes(graph)
        branches = map(lambda entity: nx.descendants(self.dag, entity), direct_root_nodes)
        branches = set().union(*branches)
        co_hyponyms = branches - self.hypernyms_and_hyponyms_and_self(entity)
        return co_hyponyms

    @lru_cache(maxsize=10000)
    def depth(self, entity, offset=1, not_exists=None):
        graph = self.dag

        if entity not in graph:
            return not_exists

        direct_root_nodes = nx.ancestors(graph, entity) & self._find_root_nodes(graph)
        if len(direct_root_nodes) == 0:
            depth = 0
        else:
            f_path_length_to_entity = lambda source: nx.shortest_path_length(graph, source, entity)
            depth = max(map(f_path_length_to_entity, direct_root_nodes))

        return depth + offset

    def hyponymy_score_slow(self, hypernym, hyponym, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        lowest_common_ancestor = nx.lowest_common_ancestor(graph, hypernym, hyponym)
        # 1) not connected
        if lowest_common_ancestor is None:
            dist = - self.depth(hypernym)
        # 2) hypernym is the ancestor of the hyponym
        elif lowest_common_ancestor == hypernym:
            dist = nx.shortest_path_length(graph, hypernym, hyponym)
        # 3) these two entities are the co-hyponym
        else:
            dist = - nx.shortest_path_length(graph, lowest_common_ancestor, hypernym)
        return dtype(dist)

    def hyponymy_score(self, hypernym, hyponym, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        ancestors_hypernym = nx.ancestors(self.dag, hypernym)
        ancestors_hyponym = nx.ancestors(self.dag, hyponym)
        ancestors_common = ancestors_hypernym.intersection(ancestors_hyponym)

        # 1) hypernym is the ancestor of the hyponym
        if hypernym in ancestors_hyponym:
            dist = nx.shortest_path_length(graph, hypernym, hyponym)
        # 2) hyponym is the ancestor of the hypernym (=reverse hyponymy)
        elif hyponym in ancestors_hypernym:
            dist = - nx.shortest_path_length(graph, hyponym, hypernym)
        # 3) not connected
        elif len(ancestors_common) == 0:
            dist = - self.depth(hypernym)
        # 4) these two entities are the co-hyponym
        elif len(ancestors_common) > 0:
            lst_path_length = (nx.shortest_path_length(graph, common, hypernym) for common in ancestors_common)
            dist = - min(lst_path_length)
        return dtype(dist)

    def lowest_common_ancestor_depth(self, hypernym, hyponym, offset: int =1, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        ancestors_hypernym = nx.ancestors(self.dag, hypernym)
        ancestors_hyponym = nx.ancestors(self.dag, hyponym)
        ancestors_common = ancestors_hypernym.intersection(ancestors_hyponym)

        # 1) hypernym is the ancestor of the hyponym: LCA is hypernym
        if hypernym in ancestors_hyponym:
            depth_lca = self.depth(entity=hypernym, offset=offset)
        # 2) hyponym is the ancestor of the hypernym (=reverse hyponymy): LCA is hyponym
        elif hyponym in ancestors_hypernym:
            depth_lca = self.depth(entity=hyponym, offset=offset)
        # 3) not connected -> LCA is empty
        elif len(ancestors_common) == 0:
            depth_lca = 0
        # 4) these two entities are the co-hyponym: LCA is the deepest co-hyponym.
        elif len(ancestors_common) > 0:
            lst_depth = (self.depth(entity=common, offset=offset) for common in ancestors_common)
            depth_lca = max(lst_depth)
        return dtype(depth_lca)

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True,
                            candidate_weight_function = None) -> List[str]:
        graph = self.dag
        if entity not in graph:
            return []

        if exclude_hypernyms:
            non_candidates = self.hypernyms_and_hyponyms_and_self(entity)
        else:
            non_candidates = self.hyponyms_and_self(entity)
        candidates = self._nodes if candidates is None else set(candidates).intersection(self._nodes)
        candidates = candidates - non_candidates

        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [next(iter(candidates))]*size

        # sampling with replacement
        # if `candidate_weight_function` is specified, then weighted sampling
        candidates = list(candidates)
        if candidate_weight_function is None:
            sampled = np.random.choice(candidates, size=size)
        else:
            vec_weights = np.fromiter(candidate_weight_function(candidates), dtype=np.float)
            vec_weights = vec_weights / np.sum(vec_weights)
            sampled = np.random.choice(candidates, size=size, p=vec_weights)

        sampled = sampled.tolist()

        return sampled

    def sample_random_hyponyms(self, entity: str,
                               candidates: Optional[Iterable[str]] = None,
                               size: int = 1, exclude_hypernyms: bool = True,
                               weighted_sampling: bool = False):
        if weighted_sampling:
            weight_function = lambda e: self.hyponym_frequency(e, not_exist=1)
        else:
            weight_function = None
        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms,
                                                             candidate_weight_function=weight_function)
        lst_ret = [(entity, hyponym, self.hyponymy_score(entity, hyponym)) for hyponym in lst_non_hyponymy_entities]

        return lst_ret

    def sample_random_hypernyms(self, entity: str,
                                candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True,
                                weighted_sampling: bool = False):

        if weighted_sampling:
            weight_function = lambda e: self.hypernym_frequency(e, not_exist=1)
        else:
            weight_function = None
        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms,
                                                             candidate_weight_function=weight_function)
        lst_ret = [(hypernym, entity, self.hyponymy_score(hypernym, entity)) for hypernym in lst_non_hyponymy_entities]

        return lst_ret

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None):
        graph = self.dag
        if (hypernym not in graph) or (hyponym not in graph):
            return not_exists

        if include_reverse_hyponymy:
            candidates = self.hyponyms(hypernym) | self.hypernyms(hypernym)
        else:
            candidates = self.hyponyms(hypernym)
        ret = hyponym in candidates

        return ret

    def sample_random_co_hyponyms(self, hypernym: str, hyponym: str, size: int = 1, break_probability: float = 0.8):
        graph = self.dag
        lst_co_hyponymy = []
        if (hypernym not in graph) or (hyponym not in graph):
            return lst_co_hyponymy
        if not nx.has_path(graph, source=hypernym, target=hyponym):
            return lst_co_hyponymy

        for _ in range(size):
            co_hyponymy_triple = self._sample_random_co_hyponymy(hypernym, hyponym, break_probability)
            if co_hyponymy_triple is not None:
                lst_co_hyponymy.append(co_hyponymy_triple)
        return lst_co_hyponymy

    def _sample_random_co_hyponymy(self, hypernym: str, hyponym: str, break_probability: float) -> Tuple[int, int, float]:
        graph = self.dag
        shortest_path = self.hypernyms(hyponym) - self.hypernyms(hypernym)
        children = [n for n in graph.successors(hypernym) if n != hyponym]
        hyponymy_score = None

        while len(children) > 0:
            node = random.choice(children)
            if node not in shortest_path:
                hyponymy_score = -1 if hyponymy_score is None else hyponymy_score - 1
                q = random.uniform(0,1)
                if q <= break_probability:
                    break

            # update children
            children = [n for n in graph.successors(node) if n != hyponym]

        if hyponymy_score is None:
            return None
        else:
            return (node, hyponym, hyponymy_score)


class WordNetTaxonomy(BasicTaxonomy):

    _SEPARATOR = "▁" # U+2581

    def __init__(self, hyponymy_dataset: Optional[HyponymyDataset] = None, synset_aware: bool = False):

        self._synset_aware = synset_aware
        # build taxonomy as for each part-of-speech tags as DAG
        dict_iter_hyponymy_pairs = defaultdict(list)
        dict_iter_trainset_pairs = defaultdict(list)
        for record in hyponymy_dataset:
            entity_type = record["pos"]

            if synset_aware:
                lemma_hyper = record["hypernym"]
                lemma_hypo = record["hyponym"]
                synset_hyper = record["synset_hypernym"]
                synset_hypo = record["synset_hyponym"]
                entity_hyper = self.synset_and_lemma_to_entity(synset_hyper, lemma_hyper)
                entity_hypo = self.synset_and_lemma_to_entity(synset_hypo, lemma_hypo)
            else:
                entity_hyper = record["hypernym"]
                entity_hypo = record["hyponym"]

            dict_iter_trainset_pairs[entity_type].append((entity_hyper, entity_hypo))
            if record["distance"] == 1.0:
                dict_iter_hyponymy_pairs[entity_type].append((entity_hyper, entity_hypo))

        self.build_directed_acyclic_graph(dict_iter_hyponymy_pairs)
        self.record_ancestors_and_descendeants(dict_iter_trainset_pairs)

    def build_directed_acyclic_graph(self, dict_iter_hyponymy_pairs: Dict[str, Iterable[Tuple[str, str]]]):
        self._dag = {}
        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            print(f"building graph. entity type: {entity_type}")
            graph = nx.DiGraph()
            graph.add_edges_from(iter_hyponymy_pairs)
            self._dag[entity_type] = graph
            # assert nx.is_directed_acyclic_graph(graph), f"failed to construct directed acyclic graph."

        self._active_entity_type = None
        self._cache_root_nodes = {}
        self._nodes = {entity_type:set(graph.nodes) for entity_type, graph in self._dag.items()}

    def record_ancestors_and_descendeants(self, dict_iter_hyponymy_pairs):
        self._trainset_ancestors = defaultdict(lambda :defaultdict(set))
        self._trainset_descendants = defaultdict(lambda :defaultdict(set))
        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            for hypernym, hyponym in iter_hyponymy_pairs:
                self._trainset_ancestors[entity_type][hyponym].add(hypernym)
                self._trainset_descendants[entity_type][hypernym].add(hyponym)

    def synset_and_lemma_to_entity(self, synset: str, lemma: str):
        return synset + self._SEPARATOR + lemma

    def entity_to_lemma(self, entity: str):
        return entity[entity.find(self._SEPARATOR)+1:]

    def entity_to_synset_and_lemma(self, entity: str):
        return entity.split("_")

    @lru_cache(1000000)
    def hypernyms(self, entity, part_of_speech):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hypernyms(entity)

    @lru_cache(1000000)
    def hyponyms(self, entity, part_of_speech):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hyponyms(entity)

    def depth(self, entity, part_of_speech, offset=1, not_exists=None):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().depth(entity, offset, not_exists)

    def hyponymy_score_slow(self, hypernym, hyponym, part_of_speech, dtype: Type = float, not_exists=None):
        raise NotImplementedError(f"you can't use this method.")
        # self.activate_entity_type(entity_type=part_of_speech)
        # return super().hyponymy_score_slow(hypernym, hyponym, dtype)

    def hyponymy_score(self, hypernym, hyponym, part_of_speech, dtype: Type = float, not_exists=None):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hyponymy_score(hypernym, hyponym, dtype)

    def sample_non_hyponymy(self, entity, part_of_speech, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True) -> List[str]:
        self.activate_entity_type(entity_type=part_of_speech)
        return super().sample_non_hyponymy(entity, candidates, size, exclude_hypernyms)

    def sample_random_hyponyms(self, hypernym, part_of_speech, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().sample_random_hyponyms(hypernym, candidates, size, exclude_hypernyms)

    def sample_random_co_hyponyms(self, hypernym: str, hyponym: str, part_of_speech: str, size: int = 1, break_probability: float = 0.5):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().sample_random_co_hyponyms(hypernym, hyponym, size, break_probability)

    def search_entities_by_lemma(self, lemma: str, part_of_speech: str):
        self.activate_entity_type(entity_type=part_of_speech)
        if self._synset_aware:
            key = self._SEPARATOR + lemma
            return {entity for entity in self.nodes if entity.endswith(key)}
        else:
            return lemma if lemma in self.nodes else {}

    @property
    def active_entity_type(self):
        return self._active_entity_type

    def activate_entity_type(self, entity_type):
        self._active_entity_type = entity_type

    @property
    def dag(self):
        return self._dag.get(self.active_entity_type, self._dag)

    @property
    def entity_types(self):
        return set(self._dag.keys())

    @property
    def trainset_ancestors(self):
        return self._trainset_ancestors.get(self.active_entity_type, self._trainset_ancestors)

    @property
    def nodes(self):
        return self._nodes.get(self.active_entity_type, self._nodes)