#!/usr/bin/env bash

DATA=EUR-Lex
MODEL=AttentionXML
#SUFFIX=300L
SUFFIX1=
SUFFIX2=head-tail

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=AttentionXML

./scripts/run_preprocess.sh $DATA $SUFFIX1 && \
./scripts/run_xml.sh $DATA $MODEL $SUFFIX2 && \

export MLFLOW_RUN_ID=$(cat run_id.txt) && \

python evaluation.py \
--results results/$MODEL-$DATA-Tree-0-labels${SUFFIX2:+-$SUFFIX2}.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
