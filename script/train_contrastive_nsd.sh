#!/bin/bash

SUBJ_ID=${1}
# USE_VIS_MASK=${2}
SPARSE_RATIO=0.2
AVG_DATA_DIR="data/NSD-processed"
CHECKPOINT_DIR='checkpoints/WAVE-models/upstream/GPT/pytorch_model.bin'
VIS_MASK_JSON=None
# if [ $USE_VIS_MASK == 'True' ]; then
#         VIS_MASK_JSON='checkpoints/WAVE-models/index_yeo7.json'

# fi

. ~/.bashrc

TRAINING_STYLE='CSM'
LR=1e-5
LOG_DIR='results/models/NSD/constrastive-fmri/'

conda activate wave
cd ..

python src/train_nsd_contrastive.py --architecture 'GPT' \
        --pretrained-model $CHECKPOINT_DIR \
        --training-style $TRAINING_STYLE \
        --training-epochs 100 \
        --validation-steps 30 \
        --per-device-training-batch-size 16 \
        --per-device-validation-batch-size 16 \
        --learning-rate $LR \
        --log-dir $LOG_DIR \
        --log-every-n-steps 100 \
        --is-prompt True \
        --wandb True \
        --l1-lambda 0 \
        --scheduler 'step' \
        --fp16 True \
        --set-seed True \
        --seed 42 \
        --sparse-ratio $SPARSE_RATIO \
        --scheduler-step-size 10000 \
        --scheduler-gamma 1 \
        --weight-decay 0.03 \
        --eval-every-n-steps 1000 \
        --subject-id sub-$SUBJ_ID \
        --only-visual False \
        --norm-embs False \
        --train-mode 'region' \
        --vis-mask-json $VIS_MASK_JSON \
        --avg-data-dir $AVG_DATA_DIR

cd script
conda deactivate