#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Dict, Tuple
from functools import lru_cache
from nltk.corpus import wordnet as wn

@lru_cache(maxsize=int(1E6))
def extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id: str,
                                                                     semantic_relation: str,
                                                                     distinct: bool = False,
                                                                     fix_synonym_distance: bool = True) -> Tuple[List[str], List[int]]:
    lst_lemma_keys = []; lst_weights = []
    lst_related_synsets = gloss_extend(o_sense=synset_id, emb_strategy=semantic_relation)
    synset_src = wn.synset(synset_id)
    for synset_rel in lst_related_synsets:
        distance = synset_src.shortest_path_distance(synset_rel)

        if fix_synonym_distance:
            # fixed implementation. it treats disconnected sense as farthest one.
            distance = 5 if distance is None else distance
        else:
            # original implementation. it wrongly evaluates synonym (=zero distance) as farthest sense.
            distance = distance if distance else 5
        weight = 1 / (1 + distance)
        for lemma in synset_rel.lemmas():
            lemma_key = lemma.key()
            if distinct and (lemma.key() in lst_lemma_keys):
                idx = lst_lemma_keys.index(lemma_key)
                lst_weights[idx] += weight
                continue
            else:
                lst_lemma_keys.append(lemma.key())
                lst_weights.append(weight)

    return lst_lemma_keys, lst_weights


def get_related(names, relation='hypernyms'):
    """
    :param names: the synset list
    :param relation: all the relations
    :return: the extended gloss list with its according synset name
    """
    related_list = []
    for name in names:
        if relation == 'hypernyms':
            wn_relation = wn.synset(name).hypernyms()
        elif relation == 'hyponyms':
            wn_relation = wn.synset(name).hyponyms()
        elif relation == 'part_holonyms':
            wn_relation = wn.synset(name).part_holonyms()
        elif relation == 'part_meronyms':
            wn_relation = wn.synset(name).part_meronyms()
        elif relation == 'member_holonyms':
            wn_relation = wn.synset(name).member_holonyms()
        elif relation == 'member_meronyms':
            wn_relation = wn.synset(name).member_meronyms()
        elif relation == 'entailments':
            wn_relation = wn.synset(name).entailments()
        elif relation == 'attributes':
            wn_relation = wn.synset(name).attributes()
        elif relation == 'also_sees':
            wn_relation = wn.synset(name).also_sees()
        elif relation == 'similar_tos':
            wn_relation = wn.synset(name).similar_tos()
        elif relation == 'causes':
            wn_relation = wn.synset(name).causes()
        elif relation == 'verb_groups':
            wn_relation = wn.synset(name).verb_groups()
        elif relation == 'substance_holonyms':
            wn_relation = wn.synset(name).substance_holonyms()
        elif relation == 'substance_meronyms':
            wn_relation = wn.synset(name).substance_meronyms()
        elif relation == 'usage_domains':
            wn_relation = wn.synset(name).usage_domains()
        elif relation == 'pertainyms':
            wn_relation = [j.synset() for j in sum([i.pertainyms() for i in wn.synset(name).lemmas()], [])]
        elif relation == 'antonyms':
            wn_relation = [j.synset() for j in sum([i.antonyms() for i in wn.synset(name).lemmas()], [])]
        else:
            wn_relation = []
            print('no such relation, process terminated.')
        related_list += wn_relation
    return related_list


def morpho_extend(synset):
    lst_morpho_synsets = list()
    lst_synonymy_lemmas = list(sum([lemma.derivationally_related_forms() for lemma in synset.lemmas()], []))
    lst_morpho_synsets += [lemma.synset() for lemma in lst_synonymy_lemmas]
    return lst_morpho_synsets

def gloss_extend(o_sense, emb_strategy) -> List[wn.synset]:
    """
    note: this is the main algorithm for relation exploitation,
    use different relations to retrieve bag-of-synset
    :param o_sense: the potential sense that is under expansion
    :param relation_list: all the available relations that a synset might have, except 'verb_group'
    :return: extended_list_gloss: the bag-of-synset
    """
    extended_list, combine_list = list(), [wn.synset(o_sense)]
    if emb_strategy in ("all-relations", "all-relations-but-hyponymy", "all-relations-but-synonymy"):
        relation_list = ['hyponyms', 'part_holonyms', 'part_meronyms', 'member_holonyms', 'antonyms',
                     'member_meronyms', 'entailments', 'attributes', 'similar_tos', 'causes', 'pertainyms',
                     'substance_holonyms', 'substance_meronyms', 'usage_domains', 'also_sees']

        if emb_strategy == "all-relations-but-hyponymy":
            relation_list.remove("hyponyms")

        if emb_strategy != "all-relations-but-synonymy":
            extended_list += morpho_extend(wn.synset(o_sense))

    elif emb_strategy == "hyponymy":
        relation_list = ['hyponyms']
    else:
        raise ValueError(f"undefined `emb_strategy` value: {emb_strategy}")

    # expand the original sense with nearby senses using all relations but hypernyms
    for index, relation in enumerate(relation_list):
        combine_list += get_related([o_sense], relation)

    # expand the original sense with in-depth hypernyms (only one branch)
    if emb_strategy != "all-relations-but-hyponymy":
        for synset in [wn.synset(o_sense)]:
            # 親語義は原則として1個だが，2個以上の場合もある
            extended_list += synset.hypernyms()

    extended_list += combine_list

    return extended_list
