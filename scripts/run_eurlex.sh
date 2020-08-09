#!/usr/bin/env bash

DATA=EUR-Lex
MODEL=FastAttentionXML
#SUFFIX=300L
SUFFIX1=
SUFFIX2=random


./scripts/run_preprocess.sh $DATA $SUFFIX1 && \
./scripts/run_xml.sh $DATA $MODEL $SUFFIX2 && \

python evaluation.py \
--results results/$MODEL-$DATA-Tree-0-labels${SUFFIX2:+-$SUFFIX2}.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
