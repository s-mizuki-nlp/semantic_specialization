#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Callable, Optional, List, Union, Tuple, Any
import numpy as np
import torch

Array_like = Union[torch.Tensor, np.ndarray]


def preprocessor_for_monosemous_entity_annotated_corpus(record: Dict[str, Any]):
    lst_words = record["words"]
    lst_entity_spans = [entity["span"] for entity in record["monosemous_entities"]]
    return lst_words, lst_entity_spans


def numpy_to_tensor(self, object: Array_like) -> torch.Tensor:
        if isinstance(object, torch.Tensor):
            return object
        elif isinstance(object, np.ndarray):
            return torch.from_numpy(object)
        else:
            raise TypeError(f"unsupported type: {type(object)}")

