#!/bin/sh

python train_projection_heads.py \
--gloss_dataset_name="SREF_basic_lemma_embeddings_without_augmentation" \
--context_dataset_name="SemCor-bert-large-cased" \
--coef_max_pool_margin_loss=0.2 \
--gloss_projection_head_name="NormRestrictedShift" \
--context_projection_head_name="COPY" \
--cfg_contrastive_learning_dataset='{"num_hard_negatives":5}' \
--use_positives_as_in_batch_negatives=True \
--gpus="" \
--batch_size=256 \
--max_epochs=15
