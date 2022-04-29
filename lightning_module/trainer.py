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
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
from torch_optimizer import RAdam
from pytorch_lightning import LightningModule

from model.loss import ContrastiveLoss
from model.loss_unsupervised import MaxPoolingMarginLoss
from custom.optimizer import AdamWithWarmup

class FrozenBERTKKWSDTaskTrainer(LightningModule):

    def __init__(self,
                 gloss_projection_head: nn.Module,
                 contrastive_loss: ContrastiveLoss,
                 optimizer_params: Dict[str, Any],
                 context_projection_head: Optional[torch.nn.Module] = None,
                 max_pool_margin_loss: Optional[MaxPoolingMarginLoss] = None,
                 coef_max_pool_margin_loss: float = 1.0,
                 supervised_alignment_loss: Optional[ContrastiveLoss] = None,
                 coef_supervised_alignment_loss: float = 1.0,
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 ):

        super().__init__()

        self._contrastive_loss = contrastive_loss

        self._gloss_projection_head = gloss_projection_head
        if context_projection_head is None:
            warnings.warn(f"gloss_projection_head will be used as context_projection_head.")
            self._context_projection_head = gloss_projection_head
        else:
            self._context_projection_head = context_projection_head

        # ToDo: implement hyper-parameter export feature on encoder when saving hyper-parameters are helpful.
        hparams = {}
        self.save_hyperparameters(hparams)

        # set loss functions
        self._contrastive_loss = contrastive_loss
        self._max_pool_margin_loss = max_pool_margin_loss
        self._coef_max_pool_margin_loss = coef_max_pool_margin_loss
        self._supervised_alignment_loss = supervised_alignment_loss
        self._coef_supervised_alignment_loss = coef_supervised_alignment_loss

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

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).to(self._device)

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

    @property
    def metrics(self) -> Dict[str, str]:
        # ToDo: update validation metrics
        map_metric_to_validation = {
            "hp/contrastive_loss":"val_cond_cpl",
            "hp/uniformity":"val_cross_entropy",
            "hp/alignment":"val_cond_cpl_vs_gt_ratio",
            "hp/gloss_context_similarity":"val_code_inclusion_probability"
        }
        return map_metric_to_validation

    def on_train_start(self) -> None:
        init_metrics = {metric_name:0 for metric_name in self.metrics.keys()}
        self.logger.log_hyperparams(params=self.hparams, metrics=init_metrics)

    def forward(self, batch):
        t_codes, t_code_probs = self._gloss_projection_head.forward(x)
        return t_codes, t_code_probs

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

    def _forward_contrastive_task(self, projection_head: nn.Module, loss_function: ContrastiveLoss,
                                  query, positive, hard_negatives, num_hard_negatives):
        _query = projection_head(query)
        _positive = projection_head(positive)
        _hard_negatives = None if hard_negatives is None else projection_head(hard_negatives)

        if _hard_negatives is None:
            loss = loss_function.forward(queries=_query, positives=_positive)
        else:
            loss = loss_function.forward(queries=_query, positives=_positive, negatives=_hard_negatives, num_negative_samples=num_hard_negatives)
        return loss

    def _forward_max_pool_margin_task(self, gloss_projection_head: nn.Module, context_projection_head: nn.Module, loss_function: MaxPoolingMarginLoss,
                                      query, targets, num_targets):
        # query: in-context entity embeddings of a text.
        _query = context_projection_head(query)
        # targets: all candidate sense (=lemma key) embeddings for the query word (=lemma&pos pair).
        _targets = gloss_projection_head(targets)

        loss = loss_function.forward(queries=_query, targets=_targets, num_target_samples=num_targets)
        return loss

    def _forward_supervised_alignment_task(self, gloss_projection_head: nn.Module, context_projection_head: nn.Module, loss_function: ContrastiveLoss,
                                           query, positive, negatives, num_negatives):
        # query: in-context entity embeddings of a text.
        _query = context_projection_head(query)
        # positive: ground-truth sense (=lemma key) embeddings for the query word.
        _positive = gloss_projection_head(positive)
        # negatives: incorrect senses (=lemma keys) embeddings for the query word.
        _negatives = gloss_projection_head(negatives)

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
        contrastive_loss = self._forward_contrastive_task(projection_head=self._gloss_projection_head, loss_function=self._contrastive_loss, **batch[task_name])

        # (optional) max-pool margin loss
        if self._max_pool_margin_loss is None:
            max_pool_margin_loss = None
        else:
            task_name = "max_pool_margin"
            assert task_name in batch, f"you must specify '{task_name}' DataLoader."
            max_pool_margin_loss = self._forward_max_pool_margin_task(gloss_projection_head=self._gloss_projection_head, context_projection_head=self._context_projection_head,
                                                                      loss_function=self._max_pool_margin_loss,
                                                                      **batch[task_name])

        # (optional) supervised alignment loss
        if self._supervised_alignment_loss is None:
            supervised_alignment_loss = None
        else:
            task_name = "supervised_alignment"
            assert task_name in batch, f"you must specify '{task_name}' DataLoader."
            supervised_alignment_loss = self._forward_supervised_alignment_task(gloss_projection_head=self._gloss_projection_head, context_projection_head=self._context_projection_head,
                                                                                loss_function=self._supervised_alignment_loss,
                                                                                **batch[task_name])

        loss = contrastive_loss
        if max_pool_margin_loss is not None:
            loss = loss + self._coef_max_pool_margin_loss * max_pool_margin_loss
        if supervised_alignment_loss is not None:
            loss = loss + self._coef_supervised_alignment_loss * supervised_alignment_loss

        dict_losses = {
            "train_loss": loss,
            "train_loss_contrastive": contrastive_loss,
            "train_loss_max_pool_margin": max_pool_margin_loss,
            "train_loss_supervised_alignment": supervised_alignment_loss
        }
        self.log_dict(dict_losses)
        return loss

    def evaluate_metrics(self, target_codes: torch.Tensor, conditional_code_probs: torch.Tensor, generated_codes: Optional[torch.Tensor] = None, eps: float = 1E-15):
        """

        @param target_codes: (n_batch, n_digits). ground-truth sense codes.
        @param conditional_code_probs: (n_batch, n_digits, n_ary). conditional probability. Pr(Y_d|y_{<d})
        @param generated_codes: (n_batch, n_digits). generated sense code (using greedy decoding).
        @param eps:
        @return:
        """
        # one-hot encoding without smoothing
        n_ary = self._gloss_projection_head.n_ary
        t_code_probs_gt = self._aux_hyponymy_score._one_hot_encoding(t_codes=target_codes, n_ary=n_ary, label_smoothing_factor=0.0)

        # code lengths
        t_code_length_gt = (target_codes != 0).sum(axis=-1).type(torch.float)
        t_soft_code_length_pred = self._aux_hyponymy_score.calc_soft_code_length(conditional_code_probs)

        # common prefix lengths using conditional probability
        t_cond_cpl = self._aux_hyponymy_score.calc_soft_lowest_common_ancestor_length(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_lca_vs_gt_ratio = t_cond_cpl / t_code_length_gt
        t_pred_vs_gt_ratio = t_soft_code_length_pred / t_code_length_gt

        # common prefix lengths using generated codes
        if generated_codes is not None:
            if generated_codes.ndim == 3:
                t_code_pred = generated_codes.argmax(dim=-1)
            else:
                t_code_pred = generated_codes
            t_gen_cpl_batch = self._aux_hyponymy_score.calc_hard_common_ancestor_length(t_code_gt=target_codes, t_code_pred=t_code_pred)
            t_gen_cpl = torch.mean(t_gen_cpl_batch)
        else:
            t_gen_cpl = 0.0

        # entailment probability
        t_prob_entail = self._aux_hyponymy_score.calc_ancestor_probability(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_prob_synonym = self._aux_hyponymy_score.calc_synonym_probability(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_prob_inclusion = t_prob_synonym + t_prob_entail

        # cross entropy
        t_cross_entropy = self._aux_cross_entropy.forward(input_code_probabilities=conditional_code_probs, target_codes=target_codes)

        # code diversity
        code_probability_divergence = torch.mean(np.log(n_ary) + torch.sum(conditional_code_probs * torch.log(conditional_code_probs + eps), axis=-1), axis=-1)

        metrics = {
            "val_cross_entropy":t_cross_entropy,
            "val_gen_cpl":t_gen_cpl,
            "val_cond_cpl":torch.mean(t_cond_cpl),
            "val_cond_cpl_vs_gt_ratio":torch.mean(t_lca_vs_gt_ratio),
            "val_code_length_mean":torch.mean(t_soft_code_length_pred),
            "val_code_inclusion_probability":torch.mean(t_prob_inclusion),
            "val_code_length_std":torch.std(t_soft_code_length_pred),
            "val_code_length_pred_vs_gt_ratio":torch.mean(t_pred_vs_gt_ratio),
            "val_code_probability_divergence":torch.mean(code_probability_divergence)
        }
        return metrics

    def validation_step(self, batch, batch_idx):

        # forward computation without back-propagation
        t_target_codes = batch["ground_truth_synset_codes"]
        # conditional probability
        _, t_code_probs = self._gloss_projection_head._predict(**batch)
        # sense code generation and its probability
        t_codes_greedy, t_code_probs_greedy = self._gloss_projection_head._encode(**batch)

        # (required) supervised loss
        loss_supervised = self._contrastive_loss.forward(target_codes=t_target_codes, input_code_probabilities=t_code_probs)

        loss = loss_supervised

        metrics = {
            "val_loss": loss
        }

        # analysis metrics
        ## based on continuous relaxation
        metrics_repr = self.evaluate_metrics(target_codes=t_target_codes, conditional_code_probs=t_code_probs, generated_codes=t_codes_greedy)
        metrics.update(metrics_repr)

        self.log_dict(metrics)

        # copy metrics to hyper parameters
        for metric_name, validation_metric_name in self.metrics.items():
            self.log(metric_name, metrics_repr[validation_metric_name])

        # return list of generated codes
        lst_codes = []
        for code in t_codes_greedy.tolist():
            lst_codes.append("-".join(map(str, code)))

        return {"generated_codes": lst_codes}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        set_codes = set()
        n_code = 0
        for output in validation_step_outputs:
            lst_codes = output["generated_codes"]
            n_code += len(lst_codes)
            set_codes.update(lst_codes)
        n_code_unique = len(set_codes)
        self.log("val_unique_code_ratio", n_code_unique / n_code)
        self.log("hp/unique_code_ratio", n_code_unique / n_code)

    def test_step(self, batch, batch_idx):
        # ToDo: call WSD evaluator
        # self._evaluator
        pass

    def on_epoch_start(self):
        pass
