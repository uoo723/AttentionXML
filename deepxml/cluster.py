#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/24
@author yrh

"""

import os

import numpy as np
from logzero import logger
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import normalize

from deepxml.data_utils import get_data, get_sparse_feature, get_word_emb

__all__ = ['build_tree_by_level']


def _get_labels_f(emb_init, train_x, train_y):
    labels_f = np.zeros((train_y.shape[1], emb_init.shape[1]))
    labels_cnt = np.zeros(train_y.shape[1], np.int32)

    for i, labels in enumerate(train_y):
        indices = np.argwhere(labels == 1)[:, 1]
        for index in indices:
            word_cnt = np.count_nonzero(train_x[i])
            labels_f[index] += np.sum(emb_init[train_x[i]], axis=0)
            labels_cnt[index] += word_cnt

    labels_f = labels_f / labels_cnt[:, None]
    return labels_f


def build_tree_by_level(
    sparse_data_x,
    sparse_data_y,
    train_x: str,
    emb_init: str,
    mlb,
    eps: float,
    max_leaf: int,
    levels: list,
    label_emb: str,
    alg: str,
    groups_path,
):
    os.makedirs(os.path.split(groups_path)[0], exist_ok=True)
    logger.info('Clustering')
    logger.info('Getting Labels Feature')

    if label_emb == 'tf-idf':
        sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
        sparse_y = mlb.transform(sparse_labels)
        labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    elif label_emb == 'glove':
        emb_init = get_word_emb(emb_init)
        train_x, train_y = get_data(train_x, sparse_data_y)
        train_y = mlb.transform(train_y)
        labels_f = _get_labels_f(emb_init, train_x, train_y)

    logger.info(F'Start Clustering {levels}')

    levels, q = [2**x for x in levels], None

    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    while q:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            logger.info(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            logger.info(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps, alg == 'random'))
        q = next_q
    logger.info('Finish Clustering')


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float,
               random: bool = False):
    n = len(labels_i)

    if random:
        partition = np.random.permutation(n)
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]

        return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    old_dis, new_dis = -10000.0, -1.0

    if type(labels_f) == csr_matrix:
        centers = labels_f[[c1, c2]].toarray()
    else:
        centers = labels_f[[c1, c2]]

    l_labels_i, r_labels_i = None, None

    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])
