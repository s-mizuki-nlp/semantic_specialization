#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import json
import argparse
from pprint import pprint

import optuna

# from train_projection_heads import main
from mock_trainer import main

gpu_id = None
env_name = None

def _parse_args():
    parser = argparse.ArgumentParser(description="distributed hyper-parameter search for {gloss,context} projection heads trainer.")
    parser.add_argument("--storage", "-s", type=str, required=True, help="storage parameter of optuna.load_study(). e.g., mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DB}")
    parser.add_argument("--study_name", type=str, required=True, help="study-name parameter of optuna.load_study(). It must be consistent among optuna workers.")
    parser.add_argument("--env_name", type=str, required=True, default=None, help="name parameter of TensorBoardLogger(). It can be handy for distinguishing experiment groups")
    parser.add_argument("--gpus", type=str, required=False, default=None, help="GPU ID used for optuna worker.")
    parser.add_argument("--debug", action="store_true", help="debug mode.")

    return parser.parse_args()


def objective(trial: optuna.Trial):
    # fixed (=not optimized) variables
    dict_args = {
        "gpus": gpu_id,
        "name": env_name,
        "gloss_dataset_name": "SREF_basic_lemma_embeddings",
        "max_epochs": 10,
        "shuffle": True
    }

    # enable/disable max-pool margin task
    context_dataset_name = trial.suggest_categorical("context_dataset_name", ["", "SemCor-bert-large-cased", "SemCor+OMSTI-bert-large-cased"])

    # similarity module for contrastive loss
    similarity_class_name = trial.suggest_categorical("similarity_class_name", ["CosineSimilarity", "DotProductSimilarity", "ArcMarginProduct"])

    # optimization
    dict_args["batch_size"] = trial.suggest_categorical("batch_size", [64,128,256,512])

    # contrastive task に関する条件付け
    cfg_contrastive_learning_dataset = {
        "semantic_relation_for_positives": "all-relations",
        "use_taxonomy_distance_for_sampling_positives": trial.suggest_categorical("use_taxonomy_distance_for_sampling_positives", [True,False]),
        "num_hard_negatives": trial.suggest_categorical("num_hard_negatives", [-1, 0, 1, 3, 5]) # 負例に用いる同形異義語の数．-1:無制限，0:なし，N(>0):N個まで
    }
    if cfg_contrastive_learning_dataset["num_hard_negatives"] == 0:
        dict_args["use_positives_as_in_batch_negatives"] = True
    else:
        dict_args["use_positives_as_in_batch_negatives"] = trial.suggest_categorical("use_positives_as_in_batch_negatives", [True, False])
    dict_args["cfg_contrastive_learning_dataset"] = cfg_contrastive_learning_dataset

    # max-pool margin task に関する条件付け．
    if context_dataset_name == "":
        dict_args["coef_max_pool_margin_loss"] = 0.0
        dict_args["cfg_max_pool_margin_loss"] = {}
    else:
        dict_args["coef_max_pool_margin_loss"] = trial.suggest_loguniform("coef_max_pool_margin_loss", low=0.1, high=10.0)
        max_margin = trial.suggest_discrete_uniform("max_margin", low=0.3, high=0.9, q=0.1)
        top_k = trial.suggest_int("top_k", low=1, high=3)
        dict_args["cfg_max_pool_margin_loss"] = {"max_margin": max_margin, "top_k": top_k}

    # gloss/context projection head
    gloss_projection_head_name = trial.suggest_categorical("gloss_projection_head_name", ["MultiLayerPerceptron", "NormRestrictedShift", "Identity"])
    context_projection_head_name = trial.suggest_categorical("context_projection_head_name", ["COPY", "SHARED", "Identity", "MultiLayerPerceptron", "NormRestrictedShift"])

    ## 条件を満たさないものはNoneを返す
    if context_dataset_name == "":
        # context projection head should be parameter-less.
        if context_projection_head_name in ("COPY","MultiLayerPerceptron", "NormRestrictedShift"):
            return 0.0
        if gloss_projection_head_name == "Identity":
            return 0.0
    else:
        # context projection head can be parameter-less.
        if (gloss_projection_head_name == "Identity") and (context_projection_head_name in ("COPY", "SHARED", "Identity")):
            return 0.0
    dict_args["gloss_projection_head_name"] = gloss_projection_head_name
    dict_args["context_projection_head_name"] = context_projection_head_name

    # gloss projection head configuration
    if gloss_projection_head_name == "Identity":
        dict_args["cfg_gloss_projection_head"] = {}
    else:
        cfg_gloss_projection_head = {}
        cfg_gloss_projection_head["n_layer"] = trial.suggest_int("n_layer", low=1, high=3, step=1)
        if gloss_projection_head_name == "NormRestrictedShift":
            cfg_gloss_projection_head["max_l2_norm_ratio"] = trial.suggest_discrete_uniform("max_l2_norm_ratio", low=0.1, high=1.0, q=0.1)
            cfg_gloss_projection_head["init_zeroes"] = True
        dict_args["cfg_gloss_projection_head"] = cfg_gloss_projection_head

    # context projection head configuration
    if context_projection_head_name in ("COPY", "SHARED", "Identity"):
        dict_args["cfg_context_projection_head"] = {}
    else:
        cfg_context_projection_head = {}
        cfg_context_projection_head["n_layer"] = trial.suggest_int("n_layer", low=1, high=3, step=1)
        if context_projection_head_name == "NormRestrictedShift":
            cfg_context_projection_head["max_l2_norm_ratio"] = trial.suggest_discrete_uniform("max_l2_norm_ratio", low=0.1, high=1.0, q=0.1)
            cfg_context_projection_head["init_zeroes"] = True
        dict_args["cfg_context_projection_head"] = cfg_context_projection_head

    # similarity module
    cfg_similarity_class = {
        "temperature": trial.suggest_loguniform("temperature", low=0.01, high=1.0)
    }
    ## arc margin product に対する条件付け
    if similarity_class_name == "ArcMarginProduct":
        cfg_similarity_class["margin"] = trial.suggest_discrete_uniform("margin", low=0.1, high=1.3, q=0.2)
    else:
        pass
    dict_args["cfg_similarity_class"] = cfg_similarity_class
    dict_args["similarity_class_name"] = similarity_class_name


    dict_args["context_dataset_name"] = context_dataset_name

    return -main(dict_args, returned_metric="hp/wsd_eval_ALL", verbose=False)

if __name__ == "__main__":

    args = _parse_args()
    gpu_id = args.gpus
    env_name = args.env_name

    if args.debug:
        study = optuna.create_study()
    else:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    study.optimize(objective, n_trials=100)
    pprint(study.best_trial)
