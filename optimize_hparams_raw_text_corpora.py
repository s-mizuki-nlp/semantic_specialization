#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import json
import math
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
        "gloss_dataset_name": "SREF_basic_lemma_embeddings_without_augmentation",
        "eval_dataset_task_name": "WSD-SemEval2007",
        "max_epochs": 15,
        "log_every_n_steps": 500,
        "shuffle": True
    }

    # always enable max-pool margin task
    context_dataset_name = "wikitext103-bert-large-cased"

    # context dataset downsampler
    cfg_context_dataset_neighbor_sense_sampler = {
        "min_freq": None,
        "max_freq": trial.suggest_categorical("max_freq", choices=[1,3,10,30,100,None]),
        "enable_random_sampling": True,
    }
    dict_args["cfg_context_dataset_neighbor_sense_sampler"] = cfg_context_dataset_neighbor_sense_sampler

    # always use cosine similarity module
    similarity_class_name = "CosineSimilarity"

    # optimization
    batch_size = 256
    dict_args["val_check_interval"] = int(1000 * 128 / batch_size)
    dict_args["max_epochs"] = 15
    dict_args["batch_size"] = batch_size

    # contrastive task に関する条件付け
    cfg_contrastive_learning_dataset = {
        "semantic_relation_for_positives": "all-relations",
        "use_taxonomy_distance_for_sampling_positives": False,
        "num_hard_negatives": 5
    }
    dict_args["use_positives_as_in_batch_negatives"] = True
    dict_args["cfg_contrastive_learning_dataset"] = cfg_contrastive_learning_dataset
    # we don't accept no negative example configuration.
    if cfg_contrastive_learning_dataset["num_hard_negatives"] == 0:
        if dict_args["use_positives_as_in_batch_negatives"] == False:
            return 0.0

    # max-pool margin task に関する条件付け．
    dict_args["coef_max_pool_margin_loss"] = trial.suggest_uniform("coef_max_pool_margin_loss", low=0.1, high=0.5)
    dict_args["cfg_max_pool_margin_loss"] = {"top_k": 1}

    # gloss/context projection head
    gloss_projection_head_name = "NormRestrictedShift"
    context_projection_head_name = "COPY"
    dict_args["gloss_projection_head_name"] = gloss_projection_head_name
    dict_args["context_projection_head_name"] = context_projection_head_name

    # gloss projection head configuration
    cfg_gloss_projection_head = {}
    cfg_gloss_projection_head["n_layer"] = 2
    cfg_gloss_projection_head["max_l2_norm_ratio"] = trial.suggest_discrete_uniform("max_l2_norm_ratio", low=0.01, high=0.02, q=0.001)
    cfg_gloss_projection_head["constraint_type"] = "element_wise"
    cfg_gloss_projection_head["init_zeroes"] = True
    cfg_gloss_projection_head["distinguish_gloss_context_embeddings"] = False
    dict_args["cfg_gloss_projection_head"] = cfg_gloss_projection_head

    # distinguish_gloss_context_embeddings is effective for "SHARED" setting.
    if context_projection_head_name == "Identity":
        if cfg_gloss_projection_head["distinguish_gloss_context_embeddings"] == True:
            return 0.0

    # context projection head configuration
    dict_args["cfg_context_projection_head"] = {}

    # similarity module
    cfg_similarity_class = {
        "temperature": 1/64,
    }
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
