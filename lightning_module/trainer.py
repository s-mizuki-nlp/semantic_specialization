#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Callable, Union, Any, Tuple
import warnings
from collections import defaultdict
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim import Adam
from torch_optimizer import RAdam
from pytorch_lightning import LightningModule

from dataset import WSDTaskDataset
from dataset.gloss_embeddings import SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset
from evaluator.wsd_knn import FrozenBERTKNNWSDTaskEvaluator

from model.loss import ContrastiveLoss, TripletLoss
from model.loss_unsupervised import MaxPoolingMarginLoss
from model.utils import pairwise_cosine_similarity, batch_pairwise_cosine_similarity
from custom.optimizer import AdamWithWarmup

class FrozenBERTWSDTaskTrainer(LightningModule):

    def __init__(self,
                 gloss_projection_head: nn.Module,
                 context_projection_head: torch.nn.Module,
                 main_loss: Union[ContrastiveLoss, TripletLoss],
                 optimizer_params: Dict[str, Any],
                 wsd_evaluation_dataset: Optional[WSDTaskDataset] = None,
                 wsd_evaluation_glosses: Optional[Union[SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset]] = None,
                 max_pool_margin_loss: Optional[MaxPoolingMarginLoss] = None,
                 coef_sense_embeddings_regularizer: float = 0.0,
                 sense_embeddings_regularizer_type: str = "cosine",
                 coef_max_pool_margin_loss: float = 1.0,
                 supervised_alignment_loss: Optional[ContrastiveLoss] = None,
                 coef_supervised_alignment_loss: float = 1.0,
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 hparams: Optional[Dict[str, Any]] = None
                 ):

        super().__init__()

        self._wsd_evaluation_dataset = wsd_evaluation_dataset
        if wsd_evaluation_dataset is not None:
            assert wsd_evaluation_glosses is not None, f"you must specify `wsd_evaluation_glosses`"
        self._wsd_evaluation_glosses = wsd_evaluation_glosses

        self._contrastive_or_triplet_loss = main_loss

        self._gloss_projection_head = gloss_projection_head
        self._context_projection_head = context_projection_head

        _hparams = {} if hparams is None else hparams
        self.save_hyperparameters(_hparams)

        # set loss functions / coefficients
        self._contrastive_or_triplet_loss = main_loss
        self._max_pool_margin_loss = max_pool_margin_loss
        self._coef_max_pool_margin_loss = coef_max_pool_margin_loss
        self._supervised_alignment_loss = supervised_alignment_loss
        self._coef_supervised_alignment_loss = coef_supervised_alignment_loss
        self._coef_sense_embeddings_regularizer = coef_sense_embeddings_regularizer
        self._sense_embeddings_regularizer_type = sense_embeddings_regularizer_type

        # set optimizers
        self._optimizer_class_name = optimizer_params.pop("class_name")
        self._optimizer_params = optimizer_params

        # set model parameter scheduler
        if model_parameter_schedulers is None:
            self._model_parameter_schedulers = {}
        else:
            self._model_parameter_schedulers = model_parameter_schedulers

        if loss_parameter_schedulers is None:
            self._loss_parameter_schedulers = {}
        else:
            self._loss_parameter_schedulers = loss_parameter_schedulers

    def _get_model_device(self):
        return (next(self._gloss_projection_head.parameters())).device

    def configure_optimizers(self):
        if self._optimizer_class_name == "Adam":
            opt = Adam(self.parameters(), **self._optimizer_params)
        elif self._optimizer_class_name == "RAdam":
            opt = RAdam(self.parameters(), **self._optimizer_params)
        elif self._optimizer_class_name == "AdamWithWarmup":
            betas = (self._optimizer_params.get("beta1", 0.9), self._optimizer_params.get("beta2", 0.999))
            eps = self._optimizer_params.get("eps", 1e-8)
            optimizer = Adam(self.parameters(), lr=0, betas=betas, eps=eps)
            opt = {
                "optimizer":optimizer,
                "lr_scheduler":{
                    "scheduler":AdamWithWarmup(optimizer=optimizer, **self._optimizer_params),
                    "interval":"step"
                }
            }
        else:
            _optimizer_class = getattr(optim, self._optimizer_class_name)
            opt = _optimizer_class(params=self.parameters(), **self._optimizer_params)
        return opt

    def on_save_checkpoint(self, checkpoint):
        device = self._get_model_device()
        if device != torch.device("cpu"):
            # convert device to cpu. it changes self._model instance itself.
            _ = self._gloss_projection_head.to(device=torch.device("cpu"))
            _ = self._context_projection_head.to(device=torch.device("cpu"))
        # save model dump
        checkpoint["gloss_projection_head_dump"] = pickle.dumps(self._gloss_projection_head)
        checkpoint["context_projection_head_dump"] = pickle.dumps(self._context_projection_head)
        # then revert back if necessary.
        if device != torch.device("cpu"):
            # revert to original device (probably cuda).
            _ = self._gloss_projection_head.to(device=device)
            _ = self._context_projection_head.to(device=device)

    @classmethod
    def load_projection_heads_from_checkpoint(cls, weights_path: str, on_gpu: bool, map_location=None, fix_model_missing_attributes: bool = True) -> Tuple[nn.Module, nn.Module]:
        gloss_projection_head = cls._load_model_from_checkpoint(model_name="gloss_projection_head", weights_path=weights_path, on_gpu=on_gpu, map_location=map_location,
                                                                fix_model_missing_attributes=fix_model_missing_attributes)
        context_projection_head = cls._load_model_from_checkpoint(model_name="context_projection_head", weights_path=weights_path, on_gpu=on_gpu, map_location=map_location,
                                                                fix_model_missing_attributes=fix_model_missing_attributes)
        return gloss_projection_head, context_projection_head

    @classmethod
    def _load_model_from_checkpoint(cls, model_name: str, weights_path: str, on_gpu: bool, map_location=None, fix_model_missing_attributes: bool = True):
        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        model = pickle.loads(checkpoint[model_name + "_dump"])
        if on_gpu:
            model = model.cuda(device=map_location)
        state_dict_key_name = "_" + model_name
        model_state_dict = {key.replace(state_dict_key_name + ".", ""):param for key, param in checkpoint["state_dict"].items() if key.startswith(state_dict_key_name)}
        model.load_state_dict(model_state_dict, strict=False)

        # fix model attributes for backward compatibility.
        if fix_model_missing_attributes:
            model = cls.fix_missing_attributes(model, model_name)

        return model

    @classmethod
    def fix_missing_attributes(cls, model, model_name: Optional[str] = None):
        # d0c797: NormRestrictedShift._constraint_type DEFAULT: element_wise
        if model.__class__.__name__ == "NormRestrictedShift":
            if getattr(model, "_constraint_type", None) is None:
                setattr(model, "_constraint_type", "element_wise")

        return model

    def _update_model_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_epochs

        for parameter_name, scheduler_function in self._model_parameter_schedulers.items():
            if scheduler_function is None:
                continue

            current_value = getattr(self._gloss_projection_head, parameter_name, None)
            if current_value is not None:
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(self._gloss_projection_head, parameter_name, new_value)

                if verbose:
                    print(f"{parameter_name}: {current_value:.2f} -> {new_value:.2f}")

    def _update_loss_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_epochs

        for loss_name, dict_property_scheduler in self._loss_parameter_schedulers.items():
            # get loss layer
            if not loss_name.startswith("_"):
                loss_name = "_" + loss_name
            loss_layer = getattr(self, loss_name, None)
            if loss_layer is None:
                continue

            # get property name and apply scheduler function
            for property_name, scheduler_function in dict_property_scheduler.items():
                if scheduler_function is None:
                    continue

                # check if property exists
                if not hasattr(loss_layer, property_name):
                    continue

                current_value = getattr(loss_layer, property_name, None)
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(loss_layer, property_name, new_value)

                if verbose:
                    print(f"{loss_name}.{property_name}: {current_value:.2f} -> {new_value:.2f}")

    def _forward_contrastive_or_triplet_task(self, projection_head: nn.Module, loss_function: Union[ContrastiveLoss, TripletLoss, None],
                                             query, positive, hard_negatives, num_hard_negatives,
                                             coef_sense_embeddings_regularizer: float = 0.0, regularizer_type: str = "cosine"):
        if loss_function is None:
            return torch.tensor(0.0, dtype=torch.float, device=query.device)

        _query = projection_head(query, is_gloss_embeddings=True)
        _positive = projection_head(positive, is_gloss_embeddings=True)
        _hard_negatives = None if hard_negatives is None else projection_head(hard_negatives, is_gloss_embeddings=True)

        if _hard_negatives is None:
            loss = loss_function.forward(queries=_query, positives=_positive)
        else:
            loss = loss_function.forward(queries=_query, positives=_positive, negatives=_hard_negatives, num_negative_samples=num_hard_negatives)

        if loss_function.__class__.__name__ == "TripletLoss":
            # multiply 100x so that similar scale to contrastive loss.
            loss = loss * 100

        if coef_sense_embeddings_regularizer > 0.0:
            if regularizer_type == "l2":
                t_diff = _query - query
                n_loss = torch.linalg.norm(t_diff, dim=-1, ord=2, keepdim=False).mean() * coef_sense_embeddings_regularizer
            elif regularizer_type == "cosine":
                n_loss = (1.0 - F.cosine_similarity(query, _query, dim=-1)).mean() * coef_sense_embeddings_regularizer
            else:
                raise ValueError(f"invalid regularizer type: {regularizer_type}")
            # cosine distance
            loss = loss + n_loss

        return loss

    def _forward_max_pool_margin_task(self, gloss_projection_head: nn.Module, context_projection_head: nn.Module, loss_function: Union[MaxPoolingMarginLoss, None],
                                      query, targets, num_targets):
        if loss_function is None:
            return torch.tensor(0.0, dtype=torch.float, device=query.device)

        # query: in-context entity embeddings of a text.
        _query = context_projection_head(query, is_gloss_embeddings=False)
        # targets: all candidate sense (=lemma key) embeddings for the query word (=lemma&pos pair).
        _targets = gloss_projection_head(targets, is_gloss_embeddings=True)

        loss = loss_function.forward(queries=_query, targets=_targets, num_target_samples=num_targets)
        return loss

    def _forward_supervised_alignment_task(self, gloss_projection_head: nn.Module, context_projection_head: nn.Module, loss_function: ContrastiveLoss,
                                           query, positive, negatives, num_negatives):
        if loss_function is None:
            return torch.tensor(0.0, dtype=torch.float, device=query.device)

        # query: in-context entity embeddings of a text.
        _query = context_projection_head(query, is_gloss_embeddings=False)
        # positive: ground-truth sense (=lemma key) embeddings for the query word.
        _positive = gloss_projection_head(positive, is_gloss_embeddings=True)
        # negatives: incorrect senses (=lemma keys) embeddings for the query word.
        _negatives = gloss_projection_head(negatives, is_gloss_embeddings=True)

        loss = loss_function.forward(queries=_query, positives=_positive, negatives=_negatives, num_negative_samples=num_negatives)
        return loss

    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):

        current_step = self.trainer.global_step / (self.trainer.max_epochs * self.trainer.num_training_batches)
        self._update_model_parameters(current_step, verbose=False)
        self._update_loss_parameters(current_step, verbose=False)

        # forward computation

        # contrastive loss
        task_name = "contrastive"
        assert task_name in batch, f"you must specify '{task_name}' DataLoader."
        main_loss = self._forward_contrastive_or_triplet_task(projection_head=self._gloss_projection_head,
                                                              loss_function=self._contrastive_or_triplet_loss,
                                                              coef_sense_embeddings_regularizer=self._coef_sense_embeddings_regularizer,
                                                              regularizer_type=self._sense_embeddings_regularizer_type,
                                                              **batch[task_name])

        # (optional) max-pool margin loss
        if self._max_pool_margin_loss is None:
            max_pool_margin_loss = 0.0
        else:
            task_name = "max_pool_margin"
            assert task_name in batch, f"you must specify '{task_name}' DataLoader."
            max_pool_margin_loss = self._forward_max_pool_margin_task(gloss_projection_head=self._gloss_projection_head, context_projection_head=self._context_projection_head,
                                                                      loss_function=self._max_pool_margin_loss,
                                                                      **batch[task_name])

        # (optional) supervised alignment loss
        if self._supervised_alignment_loss is None:
            supervised_alignment_loss = 0.0
        else:
            task_name = "supervised_alignment"
            assert task_name in batch, f"you must specify '{task_name}' DataLoader."
            supervised_alignment_loss = self._forward_supervised_alignment_task(gloss_projection_head=self._gloss_projection_head, context_projection_head=self._context_projection_head,
                                                                                loss_function=self._supervised_alignment_loss,
                                                                                **batch[task_name])

        loss = main_loss + self._coef_max_pool_margin_loss * max_pool_margin_loss + self._coef_supervised_alignment_loss * supervised_alignment_loss

        dict_losses = {
            "train_loss": loss,
            "train_loss_contrastive": main_loss,
            "train_loss_max_pool_margin": max_pool_margin_loss,
            "train_loss_supervised_alignment": supervised_alignment_loss
        }
        self.log_dict(dict_losses)
        return loss

    def on_train_start(self) -> None:
        # init_metrics = {metric_name:0.0 for metric_name in self.metrics.keys()}
        # self.logger.log_hyperparams(params=self.hparams, metrics=init_metrics)
        pass

    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):

        # contrastive loss
        _batch = batch["contrastive"]
        contrastive_or_triplet_loss = self._forward_contrastive_or_triplet_task(projection_head=self._gloss_projection_head,
                                                                                loss_function=self._contrastive_or_triplet_loss,
                                                                                coef_sense_embeddings_regularizer=0.0,
                                                                                **_batch)

        # contrastive objective related metrics: alignment, uniformity
        _query = self._gloss_projection_head.forward(_batch["query"], is_gloss_embeddings=True)
        _positive = self._gloss_projection_head.forward(_batch["positive"], is_gloss_embeddings=True)
        alignment = F.cosine_similarity(_query, _positive, dim=-1).mean()
        uniformity = pairwise_cosine_similarity(_query, reduction="mean")

        # (optional) sup. alignment loss
        if "supervised_alignment" not in batch:
            supervised_alignment_loss = 0.0
            gloss_context_similarity = 0.0
            homograph_similarity = 0.0
        else:
            _batch = batch["supervised_alignment"]
            supervised_alignment_loss = self._forward_supervised_alignment_task(gloss_projection_head=self._gloss_projection_head, context_projection_head=self._context_projection_head,
                                                                                # we use contrastive loss as an alternative.
                                                                                loss_function=self._contrastive_or_triplet_loss,
                                                                                **_batch)
            _query = self._context_projection_head.forward(_batch["query"], is_gloss_embeddings=False)
            _positive = self._gloss_projection_head.forward(_batch["positive"], is_gloss_embeddings=True)
            _negatives = self._gloss_projection_head.forward(_batch["negatives"], is_gloss_embeddings=True)

            # gloss_context_similarity: ground-truth (context, gloss) pairの類似度
            gloss_context_similarity = F.cosine_similarity(_query, _positive, dim=-1).mean()

            # homograph_similarity: 語義が異なる gloss embeddings の類似度
            # gloss embeddings = positive + negatives
            _homographs = torch.cat((_positive.unsqueeze(dim=1), _negatives), dim=1)
            _num_homographs = _batch["num_negatives"] + 1
            homograph_similarity = batch_pairwise_cosine_similarity(tensors=_homographs, num_samples=_num_homographs, reduction="mean")

        metrics = {
            "val_loss_contrastive": contrastive_or_triplet_loss,
            "val_contrastive_alignment": alignment,
            "val_contrastive_uniformity": uniformity,
            "val_loss_supervised_alignment": supervised_alignment_loss,
            "val_gloss_context_similarity": gloss_context_similarity,
            "val_homograph_similarity": homograph_similarity
        }
        self.log_dict(metrics)

        # copy metrics to hyper parameters
        for metric_name, validation_metric_name in self.metrics.items():
            self.log(metric_name, metrics[validation_metric_name])

        # return none

    @property
    def metrics(self) -> Dict[str, str]:
        map_metric_to_validation = {
            "hp/contrastive_loss": "val_loss_contrastive",
            "hp/contrastive_uniformity": "val_contrastive_uniformity",
            "hp/contrastive_alignment": "val_contrastive_alignment",
            "hp/supervised_alignment_loss": "val_loss_supervised_alignment",
            "hp/gloss_context_similarity": "val_gloss_context_similarity",
            "hp/homograph_similarity": "val_homograph_similarity"
        }
        return map_metric_to_validation

    def training_epoch_end(self, outputs) -> None:
        # Do evaluation using WSD dataset
        if self._wsd_evaluation_dataset is None:
            self.log("hp/wsd_eval", 0.0)
            return

        evaluator = FrozenBERTKNNWSDTaskEvaluator(gloss_projection_head=self._gloss_projection_head,
                                                  context_projection_head=self._context_projection_head,
                                                  evaluation_dataset=self._wsd_evaluation_dataset,
                                                  lemma_key_embeddings_dataset=self._wsd_evaluation_glosses,
                                                  similarity_metric="cosine",
                                                  device=self._get_model_device())
        dict_metrics = evaluator.evaluate()

        dict_eval_metrics = {}
        dict_eval_metrics["hp/wsd_eval_ALL"] = dict_metrics["ALL"]["f1_score_by_raganato"]
        for pos, _metrics in dict_metrics["pos_orig"].items():
            dict_eval_metrics[f"hp/wsd_eval_{pos}"] = _metrics["f1_score_by_raganato"]
        for corpus_id, _metrics in dict_metrics["corpus_id"].items():
            dict_eval_metrics[f"hp/wsd_eval_{corpus_id}"] = _metrics["f1_score_by_raganato"]
        self.log_dict(dict_eval_metrics, on_step=False, on_epoch=True)
        self.log("hp_metric", dict_eval_metrics["hp/wsd_eval_ALL"])

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        if self._gloss_projection_head.__class__.__name__ == self._context_projection_head.__class__.__name__ == "Identity":
            return
        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
