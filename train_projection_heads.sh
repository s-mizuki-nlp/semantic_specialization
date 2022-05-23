#!/bin/sh

python train_projection_heads.py \
--gloss_dataset_name="SREF_basic_lemma_embeddings" \
--context_dataset_name="" \
--coef_max_pool_margin_loss=1.0 \
--gloss_projection_head_name="NormRestrictedShift" \
--context_projection_head_name="SHARED" \
--cfg_contrastive_learning_dataset='{"num_hard_negatives":0}' \
--use_positives_as_in_batch_negatives=True \
--gpus="" \
--batch_size=256 \
--max_epochs=10
