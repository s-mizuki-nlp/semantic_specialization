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

def nullable_json_loads(value):
    value = value.replace("'","\"") if isinstance(value, str) else value
    return {} if not value else json.loads(value)


parser = argparse.ArgumentParser(description="batch trainer for {gloss,context} projection heads using BERT embeddings.")
parser.add_argument("--path_args", "-a", required=True, type=str, help="path to the argument json file.")
parser.add_argument("--repeats", "-r", required=False, type=int, default=1, help="number of repetition of the experiment.")
parser.add_argument("--name", type=str, required=True, default=None, help="name parameter of TensorBoardLogger(). It can be handy for distinguishing experiment groups")
parser.add_argument("--save_eval_metrics", required=False, type=nullable_string, default=None, help="save evaluation metrics to specified path with json format. if exists, appended.")
parser.add_argument("--gpus", type=str, required=False, default=None, help="GPU ID used for optuna worker.")
parser.add_argument("--optional_args", type=nullable_json_loads, required=False, default=None, help="optional arguments that is fed to train_projection_heads.py e.g., `{'coef_max_pool_margin_loss':1.0}`")
parser.add_argument("--verbose", action="store_true", help="output verbosity.")
args = parser.parse_args()

assert os.path.exists(args.path_args), f"argument json file not found: {args.path_args}"
with io.open(args.path_args) as ifs:
    dict_args_main = json.load(ifs)

for arg_name in "name,gpus".split(","):
    dict_args_main[arg_name] = getattr(args, arg_name)

for arg_name, arg_value in args.optional_args.items():
    dict_args_main[arg_name] = arg_value

for idx in range(args.repeats):
    main(dict_external_args=dict_args_main, returned_metric="hp/wsd_eval_ALL", verbose=args.verbose)

print("finished. good-bye.")