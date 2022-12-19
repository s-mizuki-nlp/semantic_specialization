#!/usr/bin/env python
# coding: utf-8

# # Projection Headsを訓練するスクリプト
# 
# ## 目的
# * gloss embeddings, context embeddingsを変形するprojection headsを訓練する
# 
# ## タスク設定
# * BERT encoderはfine-tuningしない．
# * gloss embeddings, context embeddingsは計算済み．
# 
# ## 目的関数
# * Contrastive Loss OR Triplet Loss
# * (optional) Max-Pooling margin Loss (候補語義を利用してgloss-context similarityを学習する)
# * (optional) supervised alignment loss (SemCor等を利用してgloss-context alignmentを学習する)
# 
# ### Contrastive Loss OR Triplet Loss
# * Gloss Embeddings: wordnet_gloss_corpus.cfg_embeddings のいずれか
# * Dataset: {xLemmaEmbeddings}Dataset -> ContrastiveLearningDataset
# * Collate Function: ContrastiveDatasetEmbeddingsCollateFunction
# 
# ### Max-Pooling margin loss
# * In-Context embeddings: sense_annotated_corpus.cfg_evaluation, sense_annotated_corpus.cfg_training, raw_text_corpus.cfg_embeddings
# * Dataset: {BERTEmbeddingsDataset(+SenseFrequencyBasedEntitySampler, EmptyFilter) -> WSDTaskDataset}, xLemmaEmbeddingsDataset
# * Downsampler: SenseFrequencyBasedEntitySampler exclusive for raw_text_corpus.cfg_embeddings
# * Collate Function: GlossContextSimilarityTaskEmbeddingsCollateFunction
# 
# ### Future works: Supervised Alignment loss
# * Sense-annotated corpus context embeddings: sense_annotated_corpus.cfg_training
# * Dataset: BERTEmbeddingsDataset -> WSDTaskDataset
# * Collate Function: SupervisedGlossContextAlignmentTaskEmbeddingsCollateFunction
# 
# ## 評価用リソース
# * WSDタスク評価用データセット: sense_annotated_corpus.cfg_evaluation["WSDEval-ALL-bert-large-cased"]

from typing import Dict, Any, Optional
import sys, io, os, json, copy, warnings
from pprint import pprint
import argparse
import inspect
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ChainDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.supporters import CombinedLoader

wd = os.path.dirname(__file__)
wd = "./" if not wd else wd
os.chdir(wd)
DEFAULT_SAVE_DIR = os.path.join(wd, "./checkpoints/")
_platform = platform.node()
if _platform == "Ubuntu-Precision-Tower-3420":
    PLATFORM_NAME = "local"
else:
    PLATFORM_NAME = _platform
print(f"platform: {PLATFORM_NAME}")

from lightning_module.trainer import FrozenBERTWSDTaskTrainer
from lightning_module import custom_collate_fn

from config_files.wsd_task import cfg_task_dataset
from config_files import sense_annotated_corpus, raw_text_corpus, wordnet_gloss_corpus

from dataset.contextualized_embeddings import BERTEmbeddingsDataset
from dataset.gloss_embeddings import SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset
from dataset.contrastive_task import ContrastiveLearningDataset
from dataset import WSDTaskDataset

from model import encoder
from model import similarity
from model.loss import ContrastiveLoss, TripletLoss
from model.loss_unsupervised import MaxPoolingMarginLoss

def _default_configs():
    dict_default_projection_head = {
        "n_layer": 2,
        "n_block": 6,
        "max_l2_norm_ratio": 0.5,
        "max_l2_norm_value": 0.5,
        "init_zeroes": True,
        "distinguish_gloss_context_embeddings": False,
        "constraint_type": "element_wise"
    }

    dict_defaults = {
        "cfg_contrastive_learning_dataset": {
            "semantic_relation_for_positives": "all-relations",
            "use_taxonomy_distance_for_sampling_positives": True,
            "num_hard_negatives": -1 # 負例に用いる同形異義語の数．-1:無制限，0:なし，N(>0):N個まで
        },
        "cfg_gloss_projection_head": dict_default_projection_head,
        "cfg_context_projection_head": dict_default_projection_head,
        "cfg_similarity_class": {
            "temperature": 0.1,
            "margin": 0.5,
        },
        "cfg_max_pool_margin_loss": {
            "similarity_module": None, # default: cosine_similarity
            "label_threshold": 0.0,
            "top_k": 1
        },
        "cfg_optimizer": {
            "lr":0.001,
            "class_name":"Adam"
        },
        "cfg_trainer": {
            # "accumulate_grad_batches":None,
            # "gradient_clip_val":1.0
            "progress_bar_refresh_rate": 20
        },
        # Raw-text corpora based context embeddings dataset filter
        "cfg_context_dataset_neighbor_sense_sampler": {
            "min_freq": None,
            "max_freq": None,
            "enable_random_sampling": False
        }
    }

    # for key, value in dict_defaults.items():
    #     dict_defaults[key] = json.dumps(value)

    return dict_defaults

def _parse_args(exclude_required_arguments: bool = False):

    def nullable_string(value):
        return None if not value else value

    def nullable_json_loads(value):
        value = value.replace("'","\"") if isinstance(value, str) else value
        return {} if not value else json.loads(value)

    default_configs = _default_configs()

    parser = argparse.ArgumentParser(description="trainer for {gloss,context} projection heads using BERT embeddings.")
    parser.add_argument("--eval_dataset_name", required=False, type=str, default="WSDEval-ALL-bert-large-cased", help="Evaluation dataset name.")
    parser.add_argument("--eval_dataset_task_name", required=False, type=str, default="WSD", choices=["WSD", "WSD-SemEval2007"], help="Evaluation dataset task name.")
    parser.add_argument("--dev_dataset_task_name", required=False, type=str, default="WSD-SemEval2007", help="Development dataset task name.")
    parser.add_argument("--gloss_dataset_name", required=False, type=str, default="SREF_basic_lemma_embeddings", help="Gloss embeddings dataset name.")
    parser.add_argument("--context_dataset_name", required=False, type=nullable_string, default=None, help="Context embeddings dataset name(s). Multiple names with comma delimiter can be specified. Specifying it enables max-pooling margin task.")
    parser.add_argument("--coef_max_pool_margin_loss", required=False, type=float, default=1.0, help="Coefficient of max-pooling margin task.")
    parser.add_argument("--sense_annotated_dataset_name", required=False, type=nullable_string, default=None, help="Sense-annotated corpus embeddings dataset name. Specifying it enables supervised alignment task.")
    parser.add_argument("--coef_supervised_alignment_loss", required=False, type=float, default=1.0, help="Coefficient of supervised alignment task.")
    parser.add_argument("--main_loss_class_name", required=False, type=str, default="ContrastiveLoss", choices=["ContrastiveLoss", "TripletLoss", "None"],
                        help="main loss class name for gloss embeddings specialization.")
    parser.add_argument("--similarity_class_name", required=False, type=str, default="CosineSimilarity", choices=["CosineSimilarity", "DotProductSimilarity", "ArcMarginProduct"],
                        help="similarity class for {contrastive, supervised alignment} tasks.")
    parser.add_argument("--use_positives_as_in_batch_negatives", required=False, type=bool, default=True, help="contrastive loss config. use positive examples as weak (=in-batch) negatives.")
    parser.add_argument("--coef_for_hard_negatives", required=False, type=float, default=1.0, help="coefficient for hard negative examples. DEFAULT: 1.0 (=uniform weighting)")
    parser.add_argument("--triplet_loss_margin", required=False, type=float, default=0.0, help="triplet loss margin. takes affect only when main loss is triplet loss.")
    parser.add_argument("--log_every_n_steps", required=False, type=int, default=200)
    parser.add_argument("--val_check_interval", required=False, type=int, default=500)
    parser.add_argument("--gpus", required=False, type=nullable_string, default=None, help="GPU device ids. e.g., `0,3,5`")
    parser.add_argument("--batch_size", required=False, type=int, default=128, help="default batch size. if specified, apllied to all tasks.")
    parser.add_argument("--batch_size_contrastive", required=False, type=int, default=None, help="contrastive task batch size.")
    parser.add_argument("--batch_size_max_pool_margin", required=False, type=int, default=None, help="max-pool margin task batch size.")
    parser.add_argument("--batch_size_supervised_alignment", required=False, type=int, default=None, help="supervised alignment task batch size.")
    parser.add_argument("--max_epochs", required=False, type=int, default=10, help="max. number of epochs.")
    parser.add_argument("--shuffle", required=False, default=True, help="shuffle trainset.")
    parser.add_argument("--num_workers", required=False, type=int, default=0, help="Not available yet.")
    parser.add_argument("--name", required=False, type=str, default=None, help=f"name of the model checkpoints. if no specified, {PLATFORM_NAME}")
    parser.add_argument("--version", required=False, type=nullable_string, default=None, help="model checkpoint version. if no specified, auto increment.")
    parser.add_argument("--save_eval_metrics", required=False, type=nullable_string, default=None, help="save evaluation metrics to specified path with json format. if exists, appended.")
    if not exclude_required_arguments:
        parser.add_argument("--gloss_projection_head_name", required=True, type=str, choices=["MultiLayerPerceptron", "NormRestrictedShift", "Identity"], help="gloss projection head class name.")
        parser.add_argument("--context_projection_head_name", required=True, type=str, choices=["MultiLayerPerceptron", "NormRestrictedShift", "Identity", "COPY", "SHARED"],
                            help="context projection head class name. SHARED: share with gloss projection head. COPY: copy initial model parameter from gloss projection head.")

    lst_config_names = ("cfg_contrastive_learning_dataset", "cfg_gloss_projection_head", "cfg_context_projection_head", "cfg_similarity_class",
                        "cfg_max_pool_margin_loss", "cfg_optimizer", "cfg_trainer", "cfg_context_dataset_neighbor_sense_sampler")
    for config_name in lst_config_names:
        parser.add_argument(f"--{config_name}", required=False, type=nullable_json_loads, default=json.dumps(default_configs[config_name]))

    args, unknown = parser.parse_known_args()

    return args

def _postprocess_args(args):
    if args.gpus is not None:
        args.gpus = list(map(int, args.gpus.split(",")))
        args.__setattr__("device", f"cuda:{args.gpus[0]}")
    else:
        args.__setattr__("device", "cpu")

    for task_name in ("contrastive","max_pool_margin","supervised_alignment"):
        attr_name = f"batch_size_{task_name}"
        if args.__dict__[attr_name] is None:
            args.__setattr__(attr_name, args.batch_size)

    # overwrite with specified value.
    print("=== overwrite default configurations ===")
    default_configs = _default_configs()
    lst_config_names = ("cfg_contrastive_learning_dataset", "cfg_gloss_projection_head", "cfg_context_projection_head", "cfg_similarity_class",
                        "cfg_max_pool_margin_loss", "cfg_optimizer", "cfg_trainer", "cfg_context_dataset_neighbor_sense_sampler")
    for config_name in lst_config_names:
        cfg_input = args.__dict__[config_name]
        cfg_default = copy.deepcopy(default_configs[config_name])
        for arg_name, value in cfg_input.items():
            default_value = cfg_default[arg_name]
            if default_value != value:
                print(f"{config_name}.{arg_name}: {default_value} -> {value}")
            cfg_default[arg_name] = value
        args.__setattr__(config_name, cfg_default)

    return args

def _update_args(args, params):
    """updates args in-place"""
    dargs = vars(args)
    dargs.update(params)

def main(dict_external_args: Optional[Dict[str, Any]] = None, returned_metric: str = "hp/wsd_eval_ALL", verbose: bool = True) -> float:
    if dict_external_args is not None:
        args = _parse_args(exclude_required_arguments=True)
        _update_args(args, dict_external_args)
    else:
        args = _parse_args()
    args = _postprocess_args(args)
    if verbose:
        pprint("==== arguments ===")
        pprint(vars(args), compact=True)

    ## Evaluation/Development Dataset
    print("loading evaluation dataset...")
    evalset_embeddings_name = args.eval_dataset_name
    evalset_embeddings = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_evaluation[evalset_embeddings_name])

    eval_dataset = WSDTaskDataset(bert_embeddings_dataset=evalset_embeddings, **cfg_task_dataset[args.eval_dataset_task_name])
    dev_dataset = WSDTaskDataset(bert_embeddings_dataset=evalset_embeddings, **cfg_task_dataset[args.dev_dataset_task_name])

    ## Gloss (Embeddings) Dataset
    gloss_dataset_name = args.gloss_dataset_name
    _cfg = wordnet_gloss_corpus.cfg_embeddings[gloss_dataset_name]
    if "kwargs_bert_embeddings_dataset" in _cfg:
        gloss_dataset = BERTLemmaEmbeddingsDataset(**_cfg)
    else:
        gloss_dataset = SREFLemmaEmbeddingsDataset(**_cfg)

    ## Contrastive Task Dataset
    contrastive_dataset = ContrastiveLearningDataset(gloss_dataset=gloss_dataset, **args.cfg_contrastive_learning_dataset)
    if verbose:
        pprint("=== contrastive task dataset ===")
        pprint(contrastive_dataset.verbose)

    ## (Optional) BERT Embeddings Dataset for self-training (previously written as max-pooling-margin) Task
    context_dataset_name = args.context_dataset_name
    if context_dataset_name is None:
        max_pool_margin_dataset = None
    elif context_dataset_name == evalset_embeddings_name:
        max_pool_margin_dataset = eval_dataset
    elif context_dataset_name == "":
        max_pool_margin_dataset = None
    else:
        lst_max_pool_margin_datasets = []
        for _context_dataset_name in context_dataset_name.split(","):
            if _context_dataset_name in sense_annotated_corpus.cfg_training:
                _context_dataset = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_training[_context_dataset_name])
                _max_pool_margin_dataset = WSDTaskDataset(bert_embeddings_dataset=_context_dataset, **cfg_task_dataset["WSD"])
            elif _context_dataset_name in raw_text_corpus.cfg_embeddings:
                _cfg = copy.deepcopy(raw_text_corpus.cfg_embeddings[_context_dataset_name])
                # setup neighbor sense downsampler
                pprint("=== Neighbor sense based context dataset sampler ===")
                pprint(args.cfg_context_dataset_neighbor_sense_sampler)
                path_sense_freq = _cfg.pop("path_sense_freq", None)
                if path_sense_freq is not None:
                    dict_filter_and_transformer = raw_text_corpus.setup_neighbor_sense_downsampler(path_sense_freq=path_sense_freq,
                                                                                                   **args.cfg_context_dataset_neighbor_sense_sampler)
                else:
                    warnings.warn(f"Skip filter because neighbor sense frequency information is not available.")
                    dict_filter_and_transformer = {}
                _context_dataset = BERTEmbeddingsDataset(**_cfg, **dict_filter_and_transformer)
                _max_pool_margin_dataset = WSDTaskDataset(bert_embeddings_dataset=_context_dataset, **cfg_task_dataset["TrainOnRawTextCorpus"])
            else:
                raise ValueError(f"invalid context dataset name: {context_dataset_name}")
            lst_max_pool_margin_datasets.append(_max_pool_margin_dataset)
        if len(lst_max_pool_margin_datasets) == 1:
            max_pool_margin_dataset = lst_max_pool_margin_datasets[0]
        else:
            max_pool_margin_dataset = ChainDataset(lst_max_pool_margin_datasets)

    if max_pool_margin_dataset is not None:
        if verbose:
            pprint("=== maximum margin task dataset ===")
            if isinstance(max_pool_margin_dataset, ChainDataset):
                for _dataset in max_pool_margin_dataset.datasets:
                    pprint(_dataset.verbose)
            else:
                pprint(max_pool_margin_dataset.verbose)

    ## (optional) BERT Embeddings Dataset for Supervised Alignment Task
    ## 事実上SemCor一択．
    sense_annotated_dataset_name = args.sense_annotated_dataset_name
    if sense_annotated_dataset_name is None:
        supervised_alignment_dataset = None
    elif sense_annotated_dataset_name in sense_annotated_corpus.cfg_training:
        sense_annotated_dataset = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_training[sense_annotated_dataset_name])
        supervised_alignment_dataset = WSDTaskDataset(bert_embeddings_dataset=sense_annotated_dataset, **cfg_task_dataset["WSD"])
    else:
        raise ValueError(f"invalid sense-annotated dataset name: {sense_annotated_dataset_name}")
    if supervised_alignment_dataset is not None:
        if verbose:
            pprint("=== supervised gloss-context alignment task dataset ===")
            pprint(supervised_alignment_dataset.verbose)

    ## Projection heads
    _encoder_classes = dict(inspect.getmembers(encoder, inspect.isclass))

    ### gloss projection head
    gloss_projection_head_name = args.gloss_projection_head_name
    if gloss_projection_head_name is None:
        gloss_projection_head = None
    else:
        assert gloss_projection_head_name in _encoder_classes, f"invalid encoder class name: {gloss_projection_head_name}"
        CLASS = _encoder_classes[gloss_projection_head_name]
        gloss_projection_head = CLASS(n_dim_in=gloss_dataset.n_dim, **args.cfg_gloss_projection_head)

    ### context projection head
    context_projection_head_name = args.context_projection_head_name
    if context_projection_head_name == "SHARED":
        warnings.warn(f"gloss_projection_head and context_projection_head will be shared.")
        context_projection_head = gloss_projection_head
    elif context_projection_head_name == "COPY":
        warnings.warn(f"context_projection_head will be initialized with same parameters as gloss_projection_head.")
        context_projection_head = copy.deepcopy(gloss_projection_head)
    else:
        assert context_projection_head_name in _encoder_classes, f"invalid encoder class name: {context_projection_head_name}"
        CLASS = _encoder_classes[context_projection_head_name]
        if context_projection_head_name == gloss_projection_head_name == "Identity":
            args.cfg_context_projection_head["assign_dummy_parameter"] = True
        context_projection_head = CLASS(n_dim_in=gloss_dataset.n_dim, **args.cfg_context_projection_head)

    _similarity_classes = dict(inspect.getmembers(similarity, inspect.isclass))
    similarity_class_name = args.similarity_class_name
    assert similarity_class_name in _similarity_classes, f"invalid similarity class name: {similarity_class_name}"
    CLASS = _similarity_classes[similarity_class_name]
    similarity_module = CLASS(**args.cfg_similarity_class)

    ## Loss functions

    ### main loss: Contrastive Loss OR Triplet Loss
    if args.main_loss_class_name == "ContrastiveLoss":
        main_loss = ContrastiveLoss(similarity_module=similarity_module,
                                    use_positives_as_in_batch_negatives=args.use_positives_as_in_batch_negatives,
                                    coef_for_hard_negatives=args.coef_for_hard_negatives)
    elif args.main_loss_class_name == "TripletLoss":
        main_loss = TripletLoss(margin=args.triplet_loss_margin, use_positives_as_in_batch_negatives=args.use_positives_as_in_batch_negatives)
    elif args.main_loss_class_name == "None":
        main_loss = None
    else:
        raise ValueError(f"invalid main_loss_class_name: {args.main_loss_class_name}")

    ### (optional) Max-Pool Margin loss
    if max_pool_margin_dataset is None:
        max_pool_margin_loss = None
    else:
        _cfg = args.cfg_max_pool_margin_loss
        if "similarity_module" not in _cfg:
            max_pool_margin_loss = MaxPoolingMarginLoss(**_cfg, similarity_module=similarity_module)
        else:
            max_pool_margin_loss = MaxPoolingMarginLoss(**_cfg)

    ### (optional) Supervised Gloss-Context Alignment loss
    if supervised_alignment_dataset is None:
        supervised_alignment_loss = None
    else:
        warnings.warn(f"We will enable supervised gloss-context alignment loss.")
        supervised_alignment_loss = ContrastiveLoss(similarity_module=similarity_module, use_positives_as_in_batch_negatives=args.use_positives_as_in_batch_negatives)

    ## DataLoader
    # * `shuffle=True` の場合は `BufferedShuffleDataset` でwrapする
    # * trainset, validationsetをそれぞれDataLoaderにする

    ### train dataloaders
    train_data_loaders = {}
    dict_task_and_datasets = {
        "contrastive": {
            "dataset": contrastive_dataset,
            "gloss_dataset": None,
            "batch_size": args.batch_size_contrastive
        },
        "max_pool_margin": {
            "dataset": max_pool_margin_dataset,
            "gloss_dataset": gloss_dataset,
            "batch_size": args.batch_size_max_pool_margin
        },
        "supervised_alignment": {
            "dataset": supervised_alignment_dataset,
            "gloss_dataset": gloss_dataset,
            "batch_size": args.batch_size_supervised_alignment
        }
    }
    for task_name, datasets in dict_task_and_datasets.items():
        if datasets["dataset"] is not None:
            train_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, shuffle=args.shuffle, device=args.device,
                                                                                **datasets)

    for task_name, data_loader in train_data_loaders.items():
        batch = next(iter(data_loader))
        print(f"{task_name}:{batch.keys()}")

    train_data_loader = CombinedLoader(train_data_loaders, mode="min_size")

    ## validation dataloaders
    val_data_loaders = {}

    ### Contrastive Task
    contrastive_dataset_val = Subset(contrastive_dataset, indices=list(range(int(len(contrastive_dataset)*0.05))))
    val_data_loaders["contrastive"] = custom_collate_fn.setup_data_loader(task_name="contrastive", dataset=contrastive_dataset_val,
                                                    shuffle=False, device=args.device, batch_size=args.batch_size_contrastive)

    ### Supervised alignment Task
    # Development setを使う．
    val_data_loaders["supervised_alignment"] = custom_collate_fn.setup_data_loader(task_name="supervised_alignment", dataset=dev_dataset, gloss_dataset=gloss_dataset,
                                                    shuffle=False, device=args.device, batch_size=args.batch_size_supervised_alignment)

    for task_name, data_loader in val_data_loaders.items():
        batch = next(iter(data_loader))
        print(f"{task_name}:{batch.keys()}")

    if len(val_data_loaders["supervised_alignment"]) <= len(val_data_loaders["contrastive"]):
        mode = "min_size"
    else:
        mode = "max_size_cycle"
    val_data_loader = CombinedLoader(val_data_loaders, mode=mode)


    ## training
    model = FrozenBERTWSDTaskTrainer(gloss_projection_head=gloss_projection_head,
                                     context_projection_head=context_projection_head,
                                     main_loss=main_loss,
                                     optimizer_params = args.cfg_optimizer,
                                     wsd_evaluation_dataset=eval_dataset,
                                     wsd_evaluation_glosses=gloss_dataset,
                                     max_pool_margin_loss=max_pool_margin_loss,
                                     coef_max_pool_margin_loss=args.coef_max_pool_margin_loss,
                                     supervised_alignment_loss=supervised_alignment_loss,
                                     coef_supervised_alignment_loss=args.coef_supervised_alignment_loss,
                                     model_parameter_schedulers=None,
                                     loss_parameter_schedulers=None,
                                     hparams=vars(args))

    platform_name = PLATFORM_NAME if args.name is None else args.name
    logger = pl_loggers.TensorBoardLogger(save_dir=DEFAULT_SAVE_DIR, name=platform_name, version=args.version, default_hp_metric=True)
    checkpoint_callback = ModelCheckpoint(filename="{epoch}", save_last=True)

    system = pl.Trainer(logger=logger, callbacks=[checkpoint_callback],
                        val_check_interval=args.val_check_interval,
                        log_every_n_steps=args.log_every_n_steps,
                        flush_logs_every_n_steps=args.log_every_n_steps,
                        max_epochs=args.max_epochs,
                        gpus=args.gpus,
                        **args.cfg_trainer
                       )
    print(f"checkpoint will be saved: {logger.log_dir}")

    _ = system.validate(model, dataloaders=val_data_loader)
    system.fit(model,
               train_dataloaders=train_data_loader,
               val_dataloaders=val_data_loader
               )

    print(f"finished: {PLATFORM_NAME}/version_{args.version}")

    path_save = args.save_eval_metrics
    if path_save is not None:
        mode = "a" if os.path.exists(path_save) else "w"
        tensor_to_item = lambda v: v.item() if torch.is_tensor(v) else v
        dict_metrics = {key: tensor_to_item(value) for key, value in system.logged_metrics.items()}
        dict_metrics["checkpoint"] = f"{platform_name}/version_{logger.version}"
        dict_metrics["index"] = f"{platform_name}"
        with io.open(path_save, mode=mode) as ofs:
            json.dump(dict_metrics, ofs)
            ofs.write("\n")

    if returned_metric is not None:
        return system.logged_metrics[returned_metric]


if __name__ == "__main__":
    main()
