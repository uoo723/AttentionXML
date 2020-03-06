#!/usr/bin/env bash

if [ $3 ]; then
  SUFFIX="-s -$3"
fi

python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 0 $SUFFIX && \
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 1 $SUFFIX && \
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 2 $SUFFIX && \
python ensemble.py -p results/$2-$1 -t 3 $SUFFIX
