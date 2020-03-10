#!/usr/bin/env bash

if [[ $3 =~ "[0-9]+L" ]]; then
  SUFFIX="-s -$3"
  INPUT_SUFFIX="-i -$3"
elif [ $3 ]; then
  SUFFIX2="-s -$3"
fi

python ensemble.py -p results/$2-$1 -t 5 ${SUFFIX2:-$SUFFIX} $INPUT_SUFFIX
