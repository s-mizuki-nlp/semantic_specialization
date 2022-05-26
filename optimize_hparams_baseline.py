#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import json
import argparse
from pprint import pprint

import optuna

from train_projection_heads import main
# from mock_trainer import main

global gpu_id
global env_name

def _parse_args():
    parser = argparse.ArgumentParser(description="distributed hyper-parameter search for {gloss,context} projection heads trainer.")
    parser.add_argument("--storage", "-s", type=str, required=True, help="storage parameter of optuna.load_study(). e.g., mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DB}")
    parser.add_argument("--n_trials", "-n", type=int, required=True, help="number of trials.")
    parser.add_argument("--study_name", type=str, required=True, help="study-name parameter of optuna.load_study(). It must be consistent among optuna workers.")
    parser.add_argument("--env_name", type=str, required=True, default=None, help="name parameter of TensorBoardLogger(). It can be handy for distinguishing experiment groups")
    parser.add_argument("--min_performance", type=float, required=False, default=0.0, help="performance below this value is regarded as zeroes. DEFAULT: 0.0")
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

    # always enable max-pool margin task
    context_dataset_name = trial.suggest_categorical("context_dataset_name", ["SemCor-bert-large-cased", "SemCor+OMSTI-bert-large-cased"])

    # always use cosine similarity module
    similarity_class_name = "CosineSimilarity"

    # optimization
    batch_size = trial.suggest_categorical("batch_size", [64,128,256,512])
    dict_args["val_check_interval"] = int(1000 * 128 / batch_size)
    if batch_size == 256:
        dict_args["max_epochs"] = 15
    elif batch_size == 512:
        dict_args["max_epochs"] = 20
    dict_args["batch_size"] = batch_size

    # contrastive task に関する条件付け
    cfg_contrastive_learning_dataset = {
        "semantic_relation_for_positives": "all-relations",
        "use_taxonomy_distance_for_sampling_positives": trial.suggest_categorical("use_taxonomy_distance_for_sampling_positives", [True,False]),
        "num_hard_negatives": trial.suggest_categorical("num_hard_negatives", [-1, 0, 1, 3, 5, 7, 9]) # 負例に用いる同形異義語の数．-1:無制限，0:なし，N(>0):N個まで
    }
    if cfg_contrastive_learning_dataset["num_hard_negatives"] == 0:
        dict_args["use_positives_as_in_batch_negatives"] = True
    else:
        dict_args["use_positives_as_in_batch_negatives"] = trial.suggest_categorical("use_positives_as_in_batch_negatives", [True, False])
    dict_args["cfg_contrastive_learning_dataset"] = cfg_contrastive_learning_dataset

    # max-pool margin task に関する条件付け．
    dict_args["coef_max_pool_margin_loss"] = trial.suggest_loguniform("coef_max_pool_margin_loss", low=0.1, high=10.0)
    max_margin = trial.suggest_discrete_uniform("max_margin", low=0.3, high=0.9, q=0.1)
    top_k = trial.suggest_int("top_k", low=1, high=3)
    dict_args["cfg_max_pool_margin_loss"] = {"max_margin": max_margin, "top_k": top_k}

    # gloss/context projection head
    gloss_projection_head_name = trial.suggest_categorical("gloss_projection_head_name", ["MultiLayerPerceptron", "NormRestrictedShift"])
    context_projection_head_name = "SHARED"
    dict_args["gloss_projection_head_name"] = gloss_projection_head_name
    dict_args["context_projection_head_name"] = context_projection_head_name

    # gloss projection head configuration
    cfg_gloss_projection_head = {}
    cfg_gloss_projection_head["n_layer"] = trial.suggest_int("n_layer", low=1, high=3, step=1)
    if gloss_projection_head_name == "NormRestrictedShift":
        cfg_gloss_projection_head["max_l2_norm_ratio"] = trial.suggest_discrete_uniform("max_l2_norm_ratio", low=0.1, high=1.0, q=0.1)
        cfg_gloss_projection_head["init_zeroes"] = True
    dict_args["cfg_gloss_projection_head"] = cfg_gloss_projection_head

    # context projection head configuration
    dict_args["cfg_context_projection_head"] = {}

    # similarity module
    cfg_similarity_class = {
        "temperature": trial.suggest_loguniform("temperature", low=0.01, high=1.0)
    }
    ## arc margin product に対する条件付け
    if similarity_class_name == "ArcMarginProduct":
        cfg_similarity_class["margin"] = trial.suggest_discrete_uniform("margin", low=0.1, high=0.5, q=0.1)
    else:
        pass
    dict_args["cfg_similarity_class"] = cfg_similarity_class
    dict_args["similarity_class_name"] = similarity_class_name

    dict_args["context_dataset_name"] = context_dataset_name

    return -max( main(dict_args, returned_metric="hp/wsd_eval_ALL", verbose=False) - args.min_performance, 0.0)

if __name__ == "__main__":

    args = _parse_args()
    gpu_id = args.gpus
    env_name = args.env_name

    if args.debug:
        study = optuna.create_study()
    else:
        study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)
    pprint(study.best_trial)
