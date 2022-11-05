#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Set, Collection, Optional
from collections import defaultdict

def compute_macro_f1_score_by_maru(dict_ground_truthes: Dict[str, Set[str]], dict_predictions: Dict[str, Set[str]],
                                   collection_evaluation_keys: Optional[Collection[str]] = None,
                                   strict: bool = False) -> Dict[str, float]:
    """
    Computes Macro-averaged F1 score following [Maru+, ACL2022].
    reference: https://github.com/SapienzaNLP/wsd-hard-benchmark/blob/main/evaluation/evaluate_macro_F1.py

    Args:
        dict_ground_truthes: ground-truth lemma keys. key: instance id, value: set of ground-truth lemma keys.
        dict_predictions: predicted lemma keys. key: instance id, value: set of predicted lemma keys.
        collection_evaluation_keys: collection of instance ids to be evaluated. if None, all instances will be evaluated.
        strict: Computes a strict macro F1 score in which the system is required to guess every gold sense when an instance is tagged with multiple gold senses
    Returns:
        Dict[str, float]: Macro-averaged precision, recall, f1_score.
    """
    tp = defaultdict(lambda: 0.)
    fp = defaultdict(lambda: 0.)
    fn = defaultdict(lambda: 0.)
    gold_keys = set()

    for instance_id in dict_ground_truthes:
        instance_gold = dict_ground_truthes[instance_id]

        if collection_evaluation_keys and instance_id not in collection_evaluation_keys:
            continue

        instance_tp, instance_fp = 0., 0.
        num_instance_predictions = 1

        if instance_id in dict_predictions:
            instance_predictions = dict_predictions[instance_id]
            num_instance_predictions = len(instance_predictions)

            for key in instance_predictions:
                if key in instance_gold:
                    instance_tp = 1. / num_instance_predictions
                else:
                    instance_fp = 1. / num_instance_predictions

            for key in instance_predictions:
                fp[key] += instance_fp

        for key in instance_gold:
            gold_keys.add(key)
            tp[key] += instance_tp
            if strict:
                if key not in instance_predictions:
                    fn[key] += 1. / num_instance_predictions
            else:
                if instance_tp == 0.:
                    fn[key] += 1. / num_instance_predictions

    avg_p = 0.
    avg_r = 0.
    avg_f1 = 0.
    total = 0

    for key in gold_keys:
        key_tp = tp[key] if key in tp else 0
        key_fp = fp[key] if key in fp else 0
        key_fn = fn[key] if key in fn else 0
        if key_tp == 0 and key_fp == 0 and key_fn == 0:
            continue

        p = key_tp / (key_tp + key_fp) if key_tp + key_fp != 0 else 0
        r = key_tp / (key_tp + key_fn) if key_tp + key_fn != 0 else 0
        f1 = 2 * (p * r) / (p + r) if p + r != 0 else 0

        avg_p += p
        avg_r += r
        avg_f1 += f1
        total += 1

    avg_p /= total
    avg_r /= total
    avg_f1 /= total

    dict_metrics = {
        "precision": avg_p,
        "recall": avg_r,
        "f1_score": avg_f1
    }

    return dict_metrics