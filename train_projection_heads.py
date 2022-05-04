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
# * Contrastive Loss
# * (optional) Max-Pooling margin Loss (候補語義を利用してgloss-context similarityを学習する)
# * (optional) supervised alignment loss (SemCor等を利用してgloss-context alignmentを学習する)
# 
# ### Contrastive Loss
# * Gloss Embeddings: wordnet_gloss_corpus.cfg_embeddings のいずれか
# * Dataset: {xLemmaEmbeddings}Dataset -> ContrastiveLearningDataset
# * Collate Function: ContrastiveDatasetEmbeddingsCollateFunction
# 
# ### Max-Pooling margin loss
# * In-Context embeddings: sense_annotated_corpus.cfg_evaluation, sense_annotated_corpus.cfg_training, monosemous_corpus.cfg_training
#     * monosemous_corpus.cfg_training は要整備
# * Dataset: {BERTEmbeddingsDataset -> WSDTaskDataset}, xLemmaEmbeddingsDataset
# * Collate Funciton: GlossContextSimilarityTaskEmbeddingsCollateFunction
# 
# ### Supervised Alignment loss
# * Sense-annotated corpus context embeddings: sense_annotated_corpus.cfg_training
#     * 単義語コーパス(monosemous_corpus.cfg_training)は使用できない．cross-entropy lossがゼロになるため．
# * Dataset: BERTEmbeddingsDataset -> WSDTaskDataset
# * Collate Function: SupervisedGlossContextAlignmentTaskEmbeddingsCollateFunction
# 
# ## 評価用リソース
# * WSDタスク評価用データセット: sense_annotated_corpus.cfg_evaluation["WSDEval-ALL-bert-large-cased"]
# 

# In[ ]:





# In[1]:


from typing import Dict, Any
import sys, io, os, json, copy
from pprint import pprint
from collections import defaultdict
import pickle, warnings
import math
import tqdm
import inspect


# In[2]:


import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, BufferedShuffleDataset
import torch.nn.functional as F


# In[3]:


import pytorch_lightning as pl


# In[4]:


wd = "/home/sakae/semantic_specialization/"
os.chdir(wd)


# In[5]:


from lightning_module.trainer import FrozenBERTWSDTaskTrainer
from lightning_module import custom_collate_fn


# In[ ]:





# In[6]:


from config_files.wsd_task import cfg_task_dataset
from config_files import sense_annotated_corpus, monosemous_corpus, wordnet_gloss_corpus


# In[7]:


from dataset.contextualized_embeddings import BERTEmbeddingsDataset
from dataset.gloss_embeddings import SREFLemmaEmbeddingsDataset, BERTLemmaEmbeddingsDataset
from dataset.contrastive_task import ContrastiveLearningDataset
from dataset import WSDTaskDataset


# In[8]:


from model import encoder
from model import similarity
from model.loss import ContrastiveLoss
from model.loss_unsupervised import MaxPoolingMarginLoss


# In[ ]:





# ## データセットの設定

# In[ ]:





# ### Evaluation/Development Dataset = WSDEval-ALL, WSDEval-SemEval2007

# In[9]:


evalset_embeddings_name = "WSDEval-ALL-bert-large-cased"
evalset_embeddings = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_evaluation[evalset_embeddings_name])

eval_dataset = WSDTaskDataset(bert_embeddings_dataset=evalset_embeddings, **cfg_task_dataset["WSD"])
dev_dataset = WSDTaskDataset(bert_embeddings_dataset=evalset_embeddings, **cfg_task_dataset["WSD-SemEval2007"])


# In[ ]:





# ### Gloss (Embeddings) Dataset

# In[10]:


gloss_dataset_name = "SREF_basic_lemma_embeddings"

gloss_dataset = SREFLemmaEmbeddingsDataset(**wordnet_gloss_corpus.cfg_embeddings[gloss_dataset_name])


# In[ ]:





# ### Contrastive Task Dataset

# In[11]:


_cfg = {
    "semantic_relation_for_positives": "all-relations",
    "use_taxonomy_distance_for_sampling_positives": True,
    "num_hard_negatives": -1 # 負例に用いる同形異義語の数．-1:無制限，0:なし，N(>0):N個まで
}
contrastive_dataset = ContrastiveLearningDataset(gloss_dataset=gloss_dataset, **_cfg)


# In[ ]:





# ### (Optional) BERT Embeddings Dataset for Max-Pooling-Margin Task

# In[12]:


# context_dataset_name = "WSDEval-ALL-bert-large-cased"
context_dataset_name = "SemCor-bert-large-cased"
# context_dataset_name = "wikitext103-subset"
# context_dataset_name = None

coef_max_pool_margin_loss = 0.1

if context_dataset_name is None:
    max_pool_margin_dataset = None
elif context_dataset_name == evalset_embeddings_name:
    max_pool_margin_dataset = eval_dataset
else:
    if context_dataset_name in sense_annotated_corpus.cfg_training:
        context_dataset = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_training[context_dataset_name])
        max_pool_margin_dataset = WSDTaskDataset(bert_embeddings_dataset=context_dataset, **cfg_task_dataset["WSD"])
    elif context_dataset_name in monosemous_corpus.cfg_training:
        context_dataset = BERTEmbeddingsDataset(**monosemous_corpus.cfg_training[context_dataset_name])
        max_pool_margin_dataset = WSDTaskDataset(bert_embeddings_dataset=context_dataset, **cfg_task_dataset["TrainOnMonosemousCorpus"])
    else:
        raise ValueError(f"invalid context dataset name: {context_dataset_name}")


# In[ ]:





# ### (optional) BERT Embeddings Dataset for Supervised Alignment Task
# * 事実上SemCor一択．

# In[37]:


sense_annotated_dataset_name = "SemCor-bert-large-cased"
sense_annotated_dataset_name = None

coef_supervised_alignment_loss = 0.0

if sense_annotated_dataset_name is None:
    supervised_alignment_dataset = None
elif sense_annotated_dataset_name in sense_annotated_corpus.cfg_training:
    sense_annotated_dataset = BERTEmbeddingsDataset(**sense_annotated_corpus.cfg_training[sense_annotated_dataset_name])
    supervised_alignment_dataset = WSDTaskDataset(bert_embeddings_dataset=context_dataset, **cfg_task_dataset["WSD"])
else:
    raise ValueError(f"invalid sense-annotated dataset name: {sense_annotated_dataset_name}")


# In[ ]:





# ## Projection headsの設定

# In[14]:


_encoder_classes = dict(inspect.getmembers(encoder, inspect.isclass))


# ### gloss projection head

# In[15]:


gloss_projection_head_name = "MultiLayerPerceptron" # MultiLayerPerceptron,NormRestrictedShift,Identity
cfg_gloss_projection_head = {
    "n_dim_in": gloss_dataset.n_dim,
    "n_dim_out": gloss_dataset.n_dim,
    "n_dim_hidden": gloss_dataset.n_dim,
    "n_layer": 2,
    "max_l2_norm_value":0.01
}

if gloss_projection_head_name is None:
    gloss_projection_head = None
else:
    assert gloss_projection_head_name in _encoder_classes, f"invalid encoder class name: {gloss_projection_head_name}"
    CLASS = _encoder_classes[gloss_projection_head_name]
    gloss_projection_head = CLASS(**cfg_gloss_projection_head)


# In[16]:


context_projection_head_name = None
# context_projection_head_name = "Identity"
cfg_context_projection_head = {
    "n_dim_in": gloss_dataset.n_dim,
    "n_dim_out": gloss_dataset.n_dim,
    "n_dim_hidden": gloss_dataset.n_dim,
    "n_layer": 2
}

if context_projection_head_name is None:
    warnings.warn(f"gloss_projection_head and context_projection_head will be shared.")
    context_projection_head = gloss_projection_head
else:
    assert context_projection_head_name in _encoder_classes, f"invalid encoder class name: {context_projection_head_name}"
    
    CLASS = _encoder_classes[context_projection_head_name]
    if context_projection_head_name == gloss_projection_head_name == "Identity":
        cfg_context_projection_head["assign_dummy_parameter"] = True
    context_projection_head = CLASS(**cfg_context_projection_head)


# In[ ]:





# ## 目的関数の設定

# In[17]:


_similarity_classes = dict(inspect.getmembers(similarity, inspect.isclass))


# ### Similarity metric

# In[18]:


SIMILARITY_CLASS_NAME = "CosineSimilarity" # CosineSimilarity,DotProductSimilarity,ArcMarginProduct
_cfg_similarity = {
    "temperature": 0.1
}

assert SIMILARITY_CLASS_NAME in _similarity_classes, f"invalid similarity class name: {SIMILARITY_CLASS_NAME}"
CLASS = _similarity_classes[SIMILARITY_CLASS_NAME]
similarity_module = CLASS(**_cfg_similarity)


# In[ ]:





# ### Loss functions

# * Contrastive Loss

# In[19]:


_cfg_contrastive_loss = {
    "use_positives_as_in_batch_negatives": True
}
contrastive_loss = ContrastiveLoss(similarity_module=similarity_module, **_cfg_contrastive_loss)


# * (optional) Max-Pool Margin loss

# In[20]:


if max_pool_margin_dataset is None:
    max_pool_margin_loss = None
else:
    _cfg_max_pool_margin_loss = {
        "similarity_module": None,
        "max_margin": 0.8
    }
    if "similarity_module" not in _cfg_max_pool_margin_loss:
        _cfg_max_pool_margin_loss["similarity_module"] = similarity_module
    max_pool_margin_loss = MaxPoolingMarginLoss(**_cfg_max_pool_margin_loss)


# * (optional) Supervised Gloss-Context Alignment loss

# In[21]:


if supervised_alignment_dataset is None:
    supervised_alignment_loss = None
else:
    warnings.warn(f"We will enable supervised gloss-context alignment loss. contrastive_loss and supervised_alignment_loss will be shared.")
    supervised_alignment_loss = contrastive_loss


# In[ ]:





# ### 実験条件

# In[22]:


import platform
platform = platform.node()
print(f"platform: {platform}")

if platform == "Ubuntu-Precision-Tower-3420":
    log_every_n_steps = 200
    val_check_interval = 500
    gpus = None
    num_workers = 0
    DEFAULT_SAVE_DIR = os.path.join(wd, "./checkpoints/")
    env_name = "local"
elif platform == "musa":
    log_every_n_steps = 50
    val_check_interval = 2000
    gpus = [3]
    num_workers = 0
    DEFAULT_SAVE_DIR = os.path.join(wd, "./checkpoints/")
    env_name = "musa"
else:
    log_every_n_steps = None
    
if gpus is None:
    device = "cpu"
else:
    device = f"cuda:{gpus[0]}"
print(f"device: {device}")


# In[ ]:





# ### 最適化

# In[23]:


batch_sizes = {
    "contrastive": 128,
    "max_pool_margin": 128,
    "supervised_alignment": 128
}
    
n_epoch = 5
shuffle = False
cfg_optimization = {
    "optimizer": {
        "lr":0.001,
        "class_name":"Adam"
    },
    "trainer": {
        # "accumulate_grad_batches":None,
        # "gradient_clip_val":1.0
    }
}


# In[ ]:





# ## DataLoader
# * `shuffle=True` の場合は `BufferedShuffleDataset` でwrapする
# * trainset, validationsetをそれぞれDataLoaderにする

# In[ ]:





# ### train dataloaders

# In[24]:


train_data_loaders = {}


# * Contrastive Task

# In[25]:


task_name = "contrastive"
train_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, dataset=contrastive_dataset, 
                                                  shuffle=shuffle, device=device, batch_size=batch_sizes[task_name])


# In[ ]:





# * (optional) Max-Pool Margin loss

# In[26]:


if max_pool_margin_dataset is not None:
    task_name = "max_pool_margin"
    train_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, 
                                                dataset=max_pool_margin_dataset, gloss_dataset=gloss_dataset,
                                                shuffle=shuffle, device=device, batch_size=batch_sizes[task_name])


# In[ ]:





# * (optional) Supervised Gloss-Context Alignment loss

# In[27]:


if supervised_alignment_dataset is not None:
    task_name = "supervised_alignment"
    train_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, 
                                                dataset=supervised_alignment_dataset, gloss_dataset=gloss_dataset,
                                                shuffle=shuffle, device=device, batch_size=batch_sizes[task_name])


# In[28]:


for task_name, data_loader in train_data_loaders.items():
    batch = next(iter(data_loader))
    print(f"{task_name}:{batch.keys()}")


# In[ ]:





# ### validation dataloaders

# In[29]:


val_data_loaders = {}


# * Contrastive Task

# In[30]:


contrastive_dataset_val = Subset(contrastive_dataset, indices=list(range(int(len(contrastive_dataset)*0.05))))

task_name = "contrastive"
val_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, dataset=contrastive_dataset_val, 
                                                shuffle=False, device=device, batch_size=batch_sizes[task_name])


# * Supervised alignment Task
# * Development setを使う．

# In[31]:


task_name = "supervised_alignment"
val_data_loaders[task_name] = custom_collate_fn.setup_data_loader(task_name=task_name, dataset=dev_dataset, gloss_dataset=gloss_dataset,
                                                shuffle=False, device=device, batch_size=batch_sizes[task_name])


# In[32]:


for task_name, data_loader in val_data_loaders.items():
    batch = next(iter(data_loader))
    print(f"{task_name}:{batch.keys()}")


# In[ ]:





# ## training
# * code memorization experiment

# In[33]:


from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


# In[34]:


pprint(cfg_optimization, compact=True)


# In[38]:


model = FrozenBERTWSDTaskTrainer(gloss_projection_head=gloss_projection_head,
                                 context_projection_head=context_projection_head,
                                 contrastive_loss=contrastive_loss,
                                 optimizer_params = cfg_optimization["optimizer"],
                                 wsd_evaluation_dataset=eval_dataset,
                                 wsd_evaluation_glosses=gloss_dataset,
                                 max_pool_margin_loss=max_pool_margin_loss,
                                 coef_max_pool_margin_loss=coef_max_pool_margin_loss,
                                 supervised_alignment_loss=supervised_alignment_loss,
                                 coef_supervised_alignment_loss=coef_supervised_alignment_loss)


# In[39]:


model.hparams


# In[40]:


logger = pl_loggers.TensorBoardLogger(save_dir=DEFAULT_SAVE_DIR, name=env_name, version=1, default_hp_metric=False)
checkpoint_callback = ModelCheckpoint(filename="{epoch}", save_last=True)

system = pl.Trainer(logger = logger, callbacks=[checkpoint_callback],
                    # limit_train_batches=5, limit_val_batches=2, num_sanity_val_steps=1,
                    val_check_interval=val_check_interval,
                    log_every_n_steps=log_every_n_steps,
                    flush_logs_every_n_steps=log_every_n_steps,
                    max_epochs=n_epoch,
                    multiple_trainloader_mode="max_size_cycle",
                    gpus=gpus,
                    **cfg_optimization["trainer"]
                   )

print(f"checkpoint will be saved: {logger.log_dir}")


# In[41]:


from pytorch_lightning.trainer.supporters import CombinedLoader


# In[ ]:


system.fit(model, train_dataloaders=train_data_loaders, val_dataloaders=CombinedLoader(val_data_loaders, mode="max_size_cycle"))


# In[ ]:




