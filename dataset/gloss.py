#!/usr/bin/env python
# -*- coding:utf-8 -*-
import io
import os
import pickle
import warnings
from collections import defaultdict
from typing import List, Optional, Union, Callable, Dict, Any

from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from torch.utils.data import Dataset

from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss


class WordNetGlossDataset(Dataset):

    def __init__(self, target_pos: List[str] = ["n","v","s","r"],
                 concat_extended_examples: bool = True,
                 lemma_surface_form_lowercase: bool = False,
                 convert_adjective_to_adjective_satellite: bool = True,
                 lst_path_extended_examples_corpus: Optional[List[str]] = None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = "",
                 verbose: bool = False):
        """
        Extended gloss dataset used in SREF [Wang and Wang, EMNLP2020]
        source: https://github.com/lwmlyy/SREF
        ref: WANG, Ming; WANG, Yinglin. A Synset Relation-enhanced Framework with a Try-again Mechanism for Word Sense Disambiguation. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. p. 6229-6240.

        @param target_pos: target part-of-speeches to generate gloss sentence.
        @param concat_extended_examples: whether if we concat extended examples collected by [Wang and Wang, EMNLP2020] or not. DEFAULT: True
        @param lemma_surface_form_lowercase: lowercase lemma surface forms (e.g., cat%1:04:00:: -> CAT). DEFAULT: True (= equivalent to nltk.corpus.wordnet.lemmas() behaviour)
        @param lst_path_extended_examples_corpus: list of the path to extended example corpura pickle files.
        @param filter_function: filter function(s) to be applied for annotated gloss sentence object.
        @param verbose: output verbosity.
        """

        super().__init__()

        if concat_extended_examples:
            assert lst_path_extended_examples_corpus is not None, f"you must specify `lst_path_extended_examples_corpus`."

            if isinstance(lst_path_extended_examples_corpus, str):
                lst_path_extended_examples_corpus = [lst_path_extended_examples_corpus]

            for path in lst_path_extended_examples_corpus:
                assert os.path.exists(path), f"invalid path specified: {path}"

        self._target_pos = target_pos
        self._lst_path_extended_examples_corpus = lst_path_extended_examples_corpus
        self._concat_extended_examples = concat_extended_examples
        self._convert_adjective_to_adjective_satellite = convert_adjective_to_adjective_satellite

        self._description = description

        if filter_function is None:
            self._filter_function = []
        elif isinstance(filter_function, list):
            self._filter_function = filter_function
        elif not isinstance(filter_function, list):
            self._filter_function = [filter_function]

        self._lemma_surface_form_lowercase = lemma_surface_form_lowercase
        self._description = description
        self._verbose = verbose

        # preload sentence object
        self._dataset = self._preload_dataset()
        self._dataset_by_lemma_pos = self._reorder_dataset_using_lemma_and_pos(dataset=self._dataset)
        self._dataset_by_lemma_sense_key = self._reorder_dataset_using_lemma_key(dataset=self._dataset)
        self._lemma_and_pos_to_lemma_keys = self._map_lemma_and_pos_to_lemma_keys(dataset=self._dataset)

    def _preload_dataset(self):
        print(f"loading dataset...")
        lst_sentences = []
        for obj_sentence in self._annotated_sentence_loader():
            lst_sentences.append(obj_sentence)
        print(f"loaded annotated sentences: {len(lst_sentences)}")

        return lst_sentences

    def _reorder_dataset_using_lemma_and_pos(self, dataset):
        dict_lemma_and_pos = defaultdict(list)
        for record in dataset:
            for entity in record["entities"]:
                tup_lemma_pos = (entity["lemma"], entity["pos"])
                dict_lemma_and_pos[tup_lemma_pos].append(record)

        return dict_lemma_and_pos

    def _reorder_dataset_using_lemma_key(self, dataset):
        dict_lemma_key = defaultdict(list)
        for record in dataset:
            for entity in record["entities"]:
                for lemma_key in entity["ground_truth_lemma_keys"]:
                    dict_lemma_key[lemma_key].append(record)

        return dict_lemma_key

    def _map_lemma_and_pos_to_lemma_keys(self, dataset):
        dict_lemma_and_pos = defaultdict(list)
        for record in dataset:
            for entity in record["entities"]:
                tup_lemma_pos = (entity["lemma"], entity["pos"])
                dict_lemma_and_pos[tup_lemma_pos].extend(entity["ground_truth_lemma_keys"])
        # drop duplicates
        for tup_lemma_pos in dict_lemma_and_pos.keys():
            dict_lemma_and_pos[tup_lemma_pos] = list(set(dict_lemma_and_pos[tup_lemma_pos]))

        return dict_lemma_and_pos

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for idx in range(len(self)):
            record = self.__getitem__(idx)
            yield record

    def __getitem__(self, item):
        record = self._dataset[item]
        flag = False
        for filter_function in self._filter_function:
            if filter_function(record):
                flag = True
                break
        if flag:
            return None
        return record

    def _synset_loader(self) -> wn.synset:
        for pos in self._target_pos:
            for synset in wn.all_synsets(pos=pos):
                yield synset

    def _extended_examples_loader(self) -> Dict[str, List[str]]:
        dict_synset_examples = {}
        for path in self._lst_path_extended_examples_corpus:
            print(f"loading extended examples corpus: {path}")
            ifs = io.open(path, mode="rb")
            dict_s = pickle.load(ifs)
            dict_synset_examples.update(dict_s)
            ifs.close()

        return dict_synset_examples

    def _annotated_sentence_loader(self):

        if self._concat_extended_examples:
            # dict_extended_examples: {"synset_id": ["example1", "example2", ...]}
            dict_extended_examples = self._extended_examples_loader()

        for synset in self._synset_loader():
            lst_lemmas = synset.lemmas()
            synset_id = synset.name()

            # lst_lemma_surfaces = list of lemma surface forms of the synset
            lst_lemma_surfaces = list(map(utils_wordnet.lemma_to_surface_form, lst_lemmas))

            # gloss: definition sentence
            tokenized_gloss = ' '.join(word_tokenize(synset.definition()))

            # concat examples from extended examples corpora
            tokenized_examples = ""
            if self._concat_extended_examples:
                if synset_id in dict_extended_examples:
                    raw_examples = ' '.join(dict_extended_examples[synset_id])
                    tokenized_examples += ' '.join(word_tokenize(raw_examples))
            else:
                pass

            # concat synset examples from WordNet.
            # sentences are re-tokenized using word_tokenize() function.
            raw_examples = ' '.join(synset.examples())
            separator = ' ' if len(raw_examples) > 0 else ''
            tokenized_examples += separator + ' '.join( word_tokenize(raw_examples) )
            # for each lemma; append all lemmas, definition sentence and example sentences (may include augmented corpora)
            for lemma in lst_lemmas:
                lemma_surface = utils_wordnet.lemma_to_surface_form(lemma)
                tokenized_gloss_sentence = lemma_surface + ' - ' + ' , '.join(lst_lemma_surfaces) + ' - ' + tokenized_gloss
                if len(tokenized_examples) > 0:
                    tokenized_gloss_sentence += ' ' + tokenized_examples
                obj_annotated_sentence = self.render_tokenized_gloss_sentence_into_annotated_sentences(
                                                    lemma=lemma,
                                                    lemma_surface_form=lemma_surface,
                                                    tokenized_sentence=tokenized_gloss_sentence)

                yield obj_annotated_sentence

    @staticmethod
    def validate_annotated_sentence(obj_annotated_sentence: Dict[str, Any], verbose: bool = False):
        assert len(obj_annotated_sentence["entities"]) > 0, f"found non-annotated sentence: {obj_annotated_sentence}"

        # validate word sequence
        expected = obj_annotated_sentence["tokenized_sentence"]
        actual = " ".join(obj_annotated_sentence["words"])
        assert expected == actual, \
            f"wrong word sequence?\nexpected: {expected}\nactual: {actual}"

        lst_words = obj_annotated_sentence["words"]
        lst_entities = obj_annotated_sentence["entities"]
        for obj_entity in lst_entities:
            # validate surface form
            entity_span = lst_words[slice(*obj_entity["span"])]
            expected = obj_entity["lemma"].lower()
            actual = "_".join(entity_span).replace("-","_").lower()
            if expected != actual:
                if verbose:
                    warnings.warn(f"wrong entity span? {expected} != {actual}")

            # validate lemma key
            expected = wn.lemma_from_key(obj_entity["ground_truth_lemma_keys"][0]).name().lower()
            actual = obj_entity["lemma"].lower()
            assert expected == actual, f"wrong lemma key: {expected} != {actual}"

    def render_tokenized_gloss_sentence_into_annotated_sentences(self, lemma: wn.lemma,
                                                                 lemma_surface_form: str,
                                                                 tokenized_sentence: str):
        if self._lemma_surface_form_lowercase:
            lemma_surface_form = lemma_surface_form.lower()

        lst_tokens = tokenized_sentence.split(" ")
        lst_lemma_surface_tokens = lemma_surface_form.split(" ")
        pos = lemma.synset().pos()
        if self._convert_adjective_to_adjective_satellite:
            pos = "s" if pos == "a" else pos

        entity = {
            "lemma": utils_wordnet_gloss.lemma_key_to_lemma_name(lemma.key()),
            "lemma_surface_form": lemma_surface_form,
            "ground_truth_lemma_keys": [lemma.key()],
            "ground_truth_synset_ids": [lemma.synset().name()],
            "pos": pos,
            "span": [0, len(lst_lemma_surface_tokens)]
        }
        lst_surfaces = []
        for token in tokenized_sentence.split(" "):
            obj_surface = {
                "surface": token,
                "lemma": token,
                "pos": None,
                "pos_orig": None
            }
            lst_surfaces.append(obj_surface)

        # sentence object
        dict_sentence = {
            "type": "gloss",
            "tokenized_sentence": tokenized_sentence,
            "words": lst_tokens,
            "entities": [entity],
            "surfaces": lst_surfaces
        }
        return dict_sentence

    def get_records_by_lemma_and_pos(self, lemma: str, pos: str) -> List[Dict[str, Any]]:
        return self._dataset_by_lemma_pos[(lemma, pos)]

    def get_records_by_lemma_key(self, lemma_key: str) -> List[Dict[str, Any]]:
        return self._dataset_by_lemma_sense_key[lemma_key]

    def get_lemma_and_pos(self):
        return list(set(self._dataset_by_lemma_pos.keys()))

    def get_lemmas(self, pos: str):
        lst_lemmas = [_lemma for _lemma, _pos in self._dataset_by_lemma_pos.keys() if _pos == pos]
        return list(set(lst_lemmas))

    def get_lemma_keys_by_lemma_and_pos(self, lemma: str, pos: str):
        return self._lemma_and_pos_to_lemma_keys[(lemma, pos)]

    def get_lemma_keys(self):
        return list(self._dataset_by_lemma_sense_key.keys())

    @property
    def verbose(self):
        ret = {
            "target_pos": self._target_pos,
            "concat_extended_examples": self._concat_extended_examples,
            "lemma_surface_form_lowercase": self._lemma_surface_form_lowercase,
            "convert_adjective_to_adjective_satellite":self._convert_adjective_to_adjective_satellite,
            "lst_path_extended_examples_corpus": self._lst_path_extended_examples_corpus,
            "__len__": self.__len__(),
            "num_lemma_and_pos": len(self.get_lemma_and_pos()),
            "num_lemma_keys": len(self.get_lemma_keys()),
            "description":self._description
        }
        return ret