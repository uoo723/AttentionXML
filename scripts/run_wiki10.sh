#!/usr/bin/env bash

DATA=Wiki10-31K
MODEL=AttentionXML
SUFFIX=300L

./scripts/run_preprocess.sh $DATA $SUFFIX && \
./scripts/run_xml.sh $DATA $MODEL $SUFFIX && \

python evaluation.py \
--results results/$MODEL-$DATA-Ensemble-labels${SUFFIX:+-$SUFFIX}.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
