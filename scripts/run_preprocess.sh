#!/usr/bin/env bash

if [ $1 == "EUR-Lex" ]; then
  TRAIN_TEXT="--text-path data/$1/train_texts.txt"
  TEST_TEXT="--text-path data/$1/test_texts.txt"
else
  TRAIN_TEXT="--text-path data/$1/train_raw_texts.txt --tokenized-path data/$1/train_texts.txt"
  TEST_TEXT="--text-path data/$1/test_raw_texts.txt --tokenized-path data/$1/test_texts.txt"
fi

if [ $2 ]; then
  MAX_LEN="--max-len $(expr match $2 '\([0-9]*\)L')"
  SUFFIX="_$2"
fi

if [ ! -f data/$1/train_texts$SUFFIX.npy ]; then
  python preprocess.py $TRAIN_TEXT --label-path data/$1/train_labels.txt --vocab-path data/$1/vocab.npy --emb-path data/$1/emb_init.npy --w2v-model data/glove.840B.300d.gensim $MAX_LEN
fi
if [ ! -f data/$1/test_texts$SUFFIX.npy ]; then
  python preprocess.py $TEST_TEXT --label-path data/$1/test_labels.txt --vocab-path data/$1/vocab.npy $MAX_LEN
fi
