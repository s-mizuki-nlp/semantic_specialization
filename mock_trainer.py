#!/usr/bin/env python
# -*- coding:utf-8 -*-

# mockup for testing hyper-parameter optimization.


from typing import Optional, Dict, Any
import os, sys, io
from pprint import pprint

from train_projection_heads import _parse_args, _update_args


def main(dict_external_args: Optional[Dict[str, Any]] = None, returned_metric: str = "hp/wsd_eval_ALL", verbose: bool = True) -> float:
    if dict_external_args is not None:
        args = _parse_args(exclude_required_arguments=True)
        _update_args(args, dict_external_args)
    else:
        args = _parse_args()
    if verbose:
        pprint("==== arguments ===")
        pprint(vars(args), compact=True)

    if args.similarity_class_name == "ArcMarginProduct":
        if args.cfg_similarity_class["temperature"] < 0.1:
            return args.coef_max_pool_margin_loss
        else:
            return 0.5
    else:
        return 0.0
