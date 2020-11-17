#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import os
import numpy as np
import joblib
import scipy as sp
import torch

from functools import reduce
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.datasets import load_svmlight_file
from gensim.models import KeyedVectors
from tqdm import tqdm
from typing import Union, Iterable, Tuple


__all__ = ['build_vocab', 'get_data', 'convert_to_binary', 'truncate_text', 'get_word_emb', 'get_mlb',
           'get_sparse_feature', 'output_res', 'mixup', 'MixUp']


def build_vocab(texts: Iterable, w2v_model: Union[KeyedVectors, str], vocab_size=500000,
                pad='<PAD>', unknown='<UNK>', sep='/SEP/', max_times=1, freq_times=1):
    if isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)
    emb_size = w2v_model.vector_size
    vocab, emb_init = [pad, unknown], [np.zeros(emb_size), np.random.uniform(-1.0, 1.0, emb_size)]
    counter = Counter(token for t in texts for token in set(t.split()))
    for word, freq in sorted(counter.items(), key=lambda x: (x[1], x[0] in w2v_model), reverse=True):
        if word in w2v_model or freq >= freq_times:
            vocab.append(word)
            # We used embedding of '.' as embedding of '/SEP/' symbol.
            word = '.' if word == sep else word
            emb_init.append(w2v_model[word] if word in w2v_model else np.random.uniform(-1.0, 1.0, emb_size))
        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab), np.asarray(emb_init)


def get_word_emb(vec_path, vocab_path=None):
    if vocab_path is not None:
        with open(vocab_path) as fp:
            vocab = {word: idx for idx, word in enumerate(fp)}
        return np.load(vec_path), vocab
    else:
        return np.load(vec_path)


def get_data(text_file, label_file=None):
    return (
        np.load(text_file, allow_pickle=True),
        np.load(label_file, allow_pickle=True) if label_file is not None else None
    )


def convert_to_binary(text_file, label_file=None, max_len=None, vocab=None, pad='<PAD>', unknown='<UNK>'):
    with open(text_file) as fp:
        texts = np.asarray([[vocab.get(word, vocab[unknown]) for word in line.split()]
                           for line in tqdm(fp, desc='Converting token to id', leave=False)])
    labels = None
    if label_file is not None:
        with open(label_file) as fp:
            labels = np.asarray([[label for label in line.split()]
                                 for line in tqdm(fp, desc='Converting labels', leave=False)])
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown]), labels


def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts


def get_mlb(mlb_path, labels=None, force=False) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path) and not force:
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    return normalize(sparse_x), np.load(label_file, allow_pickle=True) if label_file is not None else None


def output_res(output_path, name, scores, labels, suffix=''):
    os.makedirs(output_path, exist_ok=True)
    score_path = os.path.join(output_path, f'{name}-scores{suffix}.npy')
    label_path = os.path.join(output_path, f'{name}-labels{suffix}.npy')
    np.save(score_path, scores)
    np.save(label_path, labels)
    return score_path, label_path


def get_head_tail_labels(
    labels: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_cnt = Counter(label for y in labels for label in y)
    head_labels = set(map(lambda x: x[0],
                          label_cnt.most_common(int(len(label_cnt) * ratio))))
    tail_labels = set(label_cnt.keys()) - head_labels

    head_labels = np.array(list(head_labels))
    tail_labels = np.array(list(tail_labels))

    head_labels_i, tail_labels_i = get_head_tail_samples(
        head_labels, tail_labels, labels,
    )

    return head_labels, head_labels_i, tail_labels, tail_labels_i


def get_head_tail_samples(
    head_labels: np.ndarray,
    tail_labels: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    head_labels_i = set()
    tail_labels_i = set()

    for i, y in enumerate(labels):
        for label in y:
            if label in head_labels:
                head_labels_i.add(i)

            if label in tail_labels:
                tail_labels_i.add(i)

    head_labels_i = np.array(list(head_labels_i))
    tail_labels_i = np.array(list(tail_labels_i))

    return head_labels_i, tail_labels_i


def get_unique_labels(labels: np.ndarray) -> np.ndarray:
    def reducer(acc, cur):
        acc.update(cur)
        return acc

    unique_labels = reduce(reducer, labels, set())

    return np.array(list(unique_labels))


def get_splitted_samples(
    splitted_labels: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    splitted_labels_i = set()

    for i, y in enumerate(labels):
        for label in y:
            if label in splitted_labels:
                splitted_labels_i.add(i)

    splitted_labels_i = np.array(list(splitted_labels_i))

    return splitted_labels_i


def split_labels(labels: np.ndarray, num: int) -> Iterable[np.ndarray]:
    indices = np.random.permutation(len(labels))
    num_labels_per_group = len(indices) // num
    splitted_labels = []

    for i in range(num):
        start = i * num_labels_per_group
        end = start + num_labels_per_group
        index = indices[start:] if i + 1 == num else indices[start:end]
        splitted_labels.append(labels[index])

    return splitted_labels


# https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)


def mixup(x: torch.Tensor, lamda: int, indices: Iterable[int]):
    return lamda * x + (1 - lamda) * x[indices]


class MixUp:
    def __init__(self, alpha=0.2):
        self.m = torch.distributions.Beta(alpha, alpha)

    def __call__(self, train_x: torch.Tensor, train_y: torch.Tensor = None):
        lamda = self.m.sample()
        indices = torch.randperm(train_x.size(0))
        mixed_x = mixup(train_x, lamda, indices)
        ret = mixed_x
        if train_y is not None:
            mixed_y = mixup(train_y, lamda, indices)
            ret = (ret, mixed_y)
        return ret
