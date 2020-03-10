#!/usr/bin/env bash

DATA=EUR-Lex
MODEL=AttentionXML
#SUFFIX=300L
SUFFIX=5

./scripts/run_preprocess.sh $DATA $SUFFIX && \
./scripts/run_xml.sh $DATA $MODEL $SUFFIX && \

python evaluation.py \
--results results/$MODEL-$DATA-Ensemble-labels${SUFFIX:+-$SUFFIX}.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
