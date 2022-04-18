#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Set
from functools import lru_cache

import nltk
try:
    nltk.data.find("wordnet")
except Exception as e:
    print("downloading wordnet...")
    nltk.download("wordnet")

from nltk.corpus import wordnet as wn


def clean_up_lemma(str_lemma: str):
    if str_lemma.find("|") != -1:
        str_lemma = str_lemma.split("|")[0]
    if str_lemma.find("%") != -1:
        str_lemma = str_lemma.split("%")[0]
    return str_lemma

def clean_up_surface(str_surface: str):
    return str_surface.replace("\n", "")

def lemma_to_surface(str_lemma: str):
    str_lemma = clean_up_lemma(str_lemma)
    str_lemma = str_lemma.replace("_", " ")
    return str_lemma


def get_sense_key_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])

def lemma_key_to_pos(lemma_sense_key, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sense_key_type(lemma_sense_key)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sense_key_type(lemma_sense_key)]

def lemma_key_to_lemma_name(sensekey):
    return sensekey.split('%')[0]

def is_isolated_synset(synset: wn.synset, include_instance_of_relation: bool):
    if include_instance_of_relation:
        return len(synset.hypernyms()) + len(synset.instance_hypernyms()) + len(synset.hyponyms()) + len(synset.instance_hyponyms()) == 0
    else:
        return len(synset.hypernyms()) + len(synset.hyponyms()) == 0

def synset_depth(synset: wn.synset):
    return synset.max_depth()

def synset_to_lemma_keys(synset: wn.synset):
    return [lemma.key() for lemma in synset.lemmas()]

def synset_to_lemma_key_and_names(synset: wn.synset, include_instance_of_lemmas: bool = True):
    dict_ret = {}
    for lemma in synset.lemmas():
        if synset.pos() == "n" and (not include_instance_of_lemmas):
            if is_instance_of_lemma(lemma.name(), synset.pos()):
                continue
        dict_ret[lemma.key()] = lemma.name()

    return dict_ret

@lru_cache(maxsize=int(1E6))
def lemma_key_to_synset_id(lemma_key: str):
    try:
        return wn.lemma_from_key(lemma_key).synset().name()
    except:
        lemma_name = lemma_key_to_lemma_name(lemma_key)
        pos = lemma_key_to_pos(lemma_key, tagtype="short")
        for synset in wn.synsets(lemma_name, pos):
            if lemma_key in synset_to_lemma_keys(synset):
                return synset.name()
        raise ValueError(f"No synset found for key {lemma_key}")

def lemma_key_to_lemma(lemma_key: str):
    try:
        return wn.lemma_from_key(lemma_key)
    except:
        lemma_name = lemma_key_to_lemma_name(lemma_key)
        pos = lemma_key_to_pos(lemma_key, tagtype="short")
        for lemma in wn.lemmas(lemma_name, pos):
            if lemma.key() == lemma_key:
                return lemma
        raise ValueError(f"No lemma found for key {lemma_key}")

def lemma_key_to_lexname(lemma_key: str):
    lemma = lemma_key_to_lemma(lemma_key)
    return lemma.synset().lexname()

# instance-of lemmaを判定する関数
def is_instance_of_lemma(str_lemma: str, pos: str):
    lst_lemmas = wn.lemmas(str_lemma, pos)
    assert len(lst_lemmas) > 0, ValueError(f"invalid lemma: {str_lemma}|{pos}")
    # どのsynsetもinstance-of relation以外を持たない場合は instance-of lemmaだと判定する
    n_hypernyms = sum([len(lemma.synset().hypernyms()) for lemma in lst_lemmas])
    if n_hypernyms == 0:
        return True
    else:
        return False

# 単義語かどうかを判定する関数
def is_monosemous_lemma(lemma_name: str, pos: str):
    lst_lemmas = wn.lemmas(lemma_name, pos=pos)
    return len(lst_lemmas) == 1
