#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import json

import optuna

# from train_projection_heads import main
from mock_trainer import main

def objective(trial: optuna.Trial):
    dict_args = {
        "gpus": "1",
        "gloss_dataset_name": "SREF_basic_lemma_embeddings",
        "context_dataset_name": "SemCor-bert-large-cased",
    }

    context_dataset_name = trial.suggest_categorical("context_dataset_name", ["", "SemCor-bert-large-cased", ""])
    similarity_class_name = trial.suggest_categorical("similarity_class_name", ["CosineSimilarity", "DotProductSimilarity", "ArcMarginProduct"])
    temperature = trial.suggest_loguniform("temperature", low=0.01, high=1.0)
    coef_max_pool_margin_loss = trial.suggest_loguniform("coef_max_pool_margin_loss", low=0.1, high=10.0)
    max_margin = trial.suggest_float("max_margin", low=0.3, high=1.0)
    top_k = trial.suggest_int("top_k", low=1, high=3)

    cfg_similarity_class = {"temperature": temperature, "margin": 0.5}
    cfg_max_pool_margin_loss = {"max_margin": max_margin, "top_k": top_k}

    dict_args["cfg_similarity_class"] = cfg_similarity_class
    dict_args["cfg_max_pool_margin_loss"] = cfg_max_pool_margin_loss

    dict_args["similarity_class_name"] = similarity_class_name
    dict_args["coef_max_pool_margin_loss"] = coef_max_pool_margin_loss

    return -main(dict_args, returned_metric="hp/wsd_eval_ALL", verbose=False)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
