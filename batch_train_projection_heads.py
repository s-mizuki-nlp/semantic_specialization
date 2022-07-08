#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io, json
import argparse

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from train_projection_heads import main


def nullable_string(value):
    return None if not value else value

parser = argparse.ArgumentParser(description="batch trainer for {gloss,context} projection heads using BERT embeddings.")
parser.add_argument("--path_args", "-a", required=True, type=str, help="path to the argument json file.")
parser.add_argument("--repeats", "-r", required=False, type=int, default=1, help="number of repetition of the experiment.")
parser.add_argument("--env_name", type=str, required=True, default=None, help="name parameter of TensorBoardLogger(). It can be handy for distinguishing experiment groups")
parser.add_argument("--save_eval_metrics", required=False, type=nullable_string, default=None, help="save evaluation metrics to specified path with json format. if exists, appended.")
parser.add_argument("--gpus", type=str, required=False, default=None, help="GPU ID used for optuna worker.")
parser.add_argument("--verbose", action="store_true", help="output verbosity.")
args = parser.parse_args()

assert os.path.exists(args.path_args), f"argument json file not found: {args.path_args}"
with io.open(args.path_args) as ifs:
    dict_args_main = json.load(ifs)

for arg_name in "env_name,gpus".split(","):
    dict_args_main[arg_name] = getattr(args, arg_name)

for idx in range(args.repeats):
    main(dict_external_args=dict_args_main, returned_metric="hp/wsd_eval_ALL", verbose=args.verbose)

print("finished. good-bye.")