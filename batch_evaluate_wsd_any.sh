#!/bin/bash

DIR_JUPYTER_NOTEBOOK="/home/sakae/jupyter/notebook/sync_on_cloud/semantic_specialization/evaluation/"
WORK_DIR="/home/sakae/semantic_specialization"
ROOT_CHECKPOINT_DIR="${WORK_DIR}/checkpoints/"
ENV_NAME=${1}
EVALUATION_DATASET="WSDEval-ALL-bert-large-cased"

# TA=False, LCE=False
for tryagain_lce in False,False True,False False,True csi,False csi,True; do
    IFS=","; set -- $tryagain_lce;
    SUFFIX="TA-${1}_LCE-${2}"
    echo $SUFFIX
    if [ "${1}" = "csi" ]; then
	TRYAGAIN="\"${1}\""
    else
	TRYAGAIN=${1}
    fi
    
  python ./batch_evaluate_wsd.py \
	--evaluation_notebook="${DIR_JUPYTER_NOTEBOOK}/evaluate_wsd_task_using_projection_heads.ipynb" \
	--eval_dataset_name="${EVALUATION_DATASET}" \
	--env_name="${ENV_NAME}" \
	--output_summary="${WORK_DIR}/experiment_results/${ENV_NAME}_${SUFFIX}.jsonl" \
	--output_notebook="${DIR_JUPYTER_NOTEBOOK}/batch/{env_name}_version_{version_no}_${SUFFIX}.ipynb" \
	--root_checkpoint_directory="${ROOT_CHECKPOINT_DIR}" \
	--version_no="0:4" \
	--verbose \
  --evaluation_arguments="{\"gloss_embedding_name\": None, \"try_again_mechanism\":${TRYAGAIN}, \"local_context_enhancement\":${2}}"
done
