#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io, json
import argparse
from glob import glob
import regex

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

def fix_checkpoint_value(value: str, index: str):
    obj = regex.compile(r"[_a-zA-Z0-9]+\/version_[0-9]+")
    lst_found = obj.findall(value)
    if len(lst_found) > 0:
        return lst_found[0]
    else:
        obj = regex.compile(r"version_[0-9]+")
        version_no = obj.findall(value)[0]
        return f"{index}/{version_no}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="merge experiment results into single file.")
    parser.add_argument("--input_glob", "-i", required=True, type = str, default = "./experiment_results/*.json*", help = "input files with glob expression.")
    parser.add_argument("--filename_to_evaluator", required=False, type=str, default="results_from_evaluator.jsonl", help="output filename for results from evaluator.")
    parser.add_argument("--filename_to_trainer", required=False, type=str, default="results_from_trainer.jsonl", help="output filename for results from trainer.")

    args = parser.parse_args()

    for filename in [args.filename_to_trainer, args.filename_to_evaluator]:
        path = os.path.join(args.input_dir, filename)
        assert not os.path.exists(path), f"file already exists: {path}"

    lst_inputs = []
    lst_inputs.extend(glob(args.input_glob))

    assert len(lst_inputs) > 0, f"there is no experiment results."

    trainer_to_evaluator_metric = {}
    trainer_to_evaluator_metric["hp/wsd_eval_ALL"] = "ALL.f1_score_by_raganato"
    trainer_to_evaluator_metric["hp/wsd_eval_ALL_micro_f1"] = "ALL.f1_score_by_raganato"
    trainer_to_evaluator_metric["hp/wsd_eval_ALL_macro_f1"] = "ALL.f1_score"
    for pos in "NOUN,ADJ,VERB,ADV".split(","):
        trainer_metric = f"hp/wsd_eval_{pos}"
        eval_metric = f"pos_orig.{pos}.f1_score_by_raganato"
        trainer_to_evaluator_metric[trainer_metric] = eval_metric
    for corpus_id in "senseval2,senseval3,semeval2007,semeval2013,semeval2015".split(","):
        trainer_metric = f"hp/wsd_eval_{corpus_id}"
        eval_metric = f"corpus_id.{corpus_id}.f1_score_by_raganato"
        trainer_to_evaluator_metric[trainer_metric] = eval_metric

    for path in lst_inputs:
        filename = os.path.basename(path)
        index, _ = os.path.splitext(filename)

        if filename in [args.filename_to_trainer, args.filename_to_evaluator]:
            continue

        with io.open(path, mode="r") as ifs:
            lst_records = [json.loads(record.strip()) for record in ifs]

        for record in lst_records:
            is_from_trainer = True if "hp/wsd_eval_ALL" in record else False
            filename_output = args.filename_to_trainer if is_from_trainer else args.filename_to_evaluator

            # rename metrics
            if is_from_trainer:
                for trainer_metric, eval_metric in trainer_to_evaluator_metric.items():
                    record[eval_metric] = record[trainer_metric]
            # register index
            record["index"] = index
            # fix checkpoint path
            record["checkpoint"] = fix_checkpoint_value(value=record["checkpoint"], index=index)

            # save to file
            path_output = os.path.join(args.input_dir, filename_output)
            with io.open(path_output, mode="a") as ofs:
                json.dump(record, ofs)
                ofs.write("\n")

        print(f"{filename} -> {filename_output}")

    print("finished. good-bye.")
