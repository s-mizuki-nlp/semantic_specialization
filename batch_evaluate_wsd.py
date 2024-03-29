#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys, io
import warnings
import argparse
import ast
import itertools
from pprint import pprint

import papermill as pm

def _parse_slice_expression(expression: str):
    if expression.find(":") != -1:
        begin, end = (int(e) for e in expression.split(":"))
        return list(range(begin, end+1))
    else:
        return [int(expression)]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_notebook", required=True, type=str, help="path of the jupyter notebook for evaluation.")
    parser.add_argument("--output_summary", "-o", required=True, type=str, help="path of the summary output file.")
    parser.add_argument("--output_notebook", required=True, type=str, help="path of the jupyter notebook file. You can include parameter variables such as {version_no}")
    parser.add_argument("--eval_dataset_name", required=False, type=str, default="WSDEval-ALL-bert-large-cased", help="Precomputed evaluation dataset embeddings name defined in: `config_files.sense_annotated_corpus.cfg_evaluation`")
    parser.add_argument("--env_name", required=True, type=str, help="environment name which is specified when training.")
    parser.add_argument("--version_no", "-v", required=True, type=str, help="version numbers to be evaluated. ex: '28,29:31,33'")
    parser.add_argument("--root_checkpoint_directory", required=True, type=str, help="root path of the pytorch lightning checkpoint root directory. e.g., ./checkpoints/")
    parser.add_argument("--evaluation_arguments", required=False, type=str, default="{}", help="optional evaluation parameters. ex: '{\"cross_validation\":True}'")
    parser.add_argument("--device", required=False, type=str, default=None, help="device used for runnning pytorch script. e.g., 'cuda:1'")
    parser.add_argument("--append_output", action="store_true", help="append output to existing file.")
    parser.add_argument("--verbose", action="store_true", help="output verbose.")

    args = parser.parse_args()

    if os.path.exists(args.output_summary):
        if args.append_output:
            warnings.warn(f"specified path already exists. we will append results: {args.output_summary}")
        else:
            raise IOError(f"specified path already exists: {args.output_summary}")

    # parse evaluation arguments
    args.evaluation_arguments = ast.literal_eval(args.evaluation_arguments)
    # parse version numbers
    nested_lists = map(_parse_slice_expression, args.version_no.split(","))
    args.version_no = list(itertools.chain(*nested_lists))

    return args

def main():

    args = _parse_args()
    if args.verbose:
        print("evaluation configurations are as follows.")
        args_dict = {arg:getattr(args, arg) for arg in vars(args)}
        pprint(args_dict)

    for version_no in args.version_no:
        path_output_notebook = args.output_notebook.format(env_name=args.env_name, version_no=version_no, **args.evaluation_arguments)
        print(f"{version_no}: {path_output_notebook}")

        evaluation_parameters = {
            "root_checkpoint_directory": args.root_checkpoint_directory,
            "eval_dataset_name": args.eval_dataset_name,
            "env_name": args.env_name,
            "version_no":version_no,
            "path_output_batch_execution":args.output_summary,
            "batch_execution":True,
            "device": args.device,
        }
        evaluation_parameters.update(args.evaluation_arguments)
        if args.verbose:
            pprint(evaluation_parameters)

        pm.execute_notebook(
            args.evaluation_notebook,
            path_output_notebook,
            parameters=evaluation_parameters,
            progress_bar=True
        )

    print("finished. good-bye.")

if __name__ == "__main__":
    main()

