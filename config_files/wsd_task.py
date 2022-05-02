#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Optional, Union, List

from torch.utils.data import DataLoader, BufferedShuffleDataset, ChainDataset

from dataset import WSDTaskDataset, WSDTaskDatasetCollateFunction

USE_ENTITY_EMBEDDING = 0.0
USE_SENTENCE_EMBEDDING = 1.0

def WSDTaskDataLoader(dataset: Union[WSDTaskDataset, BufferedShuffleDataset],
                      batch_size: int,
                      cfg_collate_function: Dict[str, Any] = {},
                      **kwargs):
    if "has_ground_truth" not in cfg_collate_function:
        if isinstance(dataset, BufferedShuffleDataset):
            has_ground_truth = getattr(dataset.dataset, "has_ground_truth", False)
        else:
            has_ground_truth = dataset.has_ground_truth
        cfg_collate_function["has_ground_truth"] = has_ground_truth
    collate_fn = WSDTaskDatasetCollateFunction(**cfg_collate_function)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    return data_loader


cfg_task_dataset = {
    "WSD": {
        "has_ground_truth": True,
        "return_level":"entity",
        "record_entity_field_name":"entities",
        "record_entity_span_field_name":"subword_spans",
        "ground_truth_lemma_keys_field_name":"ground_truth_lemma_keys",
        "copy_field_names_from_record_to_entity":["corpus_id","document_id","sentence_id","words"],
        "return_entity_subwords_avg_vector":True
    },
    "TrainOnMonosemousCorpus": {
        "has_ground_truth": True,
        "return_level":"entity",
        "record_entity_field_name":"monosemous_entities",
        "record_entity_span_field_name":"subword_spans",
        "copy_field_names_from_record_to_entity":None,
        "return_entity_subwords_avg_vector":True
    },
    "TrainOnWordNetGlossCorpus": {
        "has_ground_truth": True,
        "return_level":"entity",
        "record_entity_field_name":"entities",
        "record_entity_span_field_name":"subword_spans",
        "ground_truth_lemma_keys_field_name":"ground_truth_lemma_keys",
        "copy_field_names_from_record_to_entity":None,
        "return_entity_subwords_avg_vector":True,
        "raise_error_on_unknown_lemma":True,
        "weighted_average_entity_embeddings_and_sentence_embedding":USE_ENTITY_EMBEDDING,
        "normalize_embeddings":False
    }
}