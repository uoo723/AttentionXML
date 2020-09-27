#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/24
@author yrh

"""

import os
import warnings

import numpy as np

from contextlib import redirect_stderr
from logzero import logger
from scipy.sparse import csc_matrix, csr_matrix

from sklearn.preprocessing import normalize

from sklearn.cluster import SpectralClustering
from sklearn.cluster.spectral import discretize
from sklearn.cluster.k_means_ import k_means
from sklearn.utils import check_array, check_random_state
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import spectral_embedding

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
    indices: np.ndarray,
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

        with redirect_stderr(None):
            sparse_y = mlb.transform(sparse_labels)

        if indices is not None:
            sparse_x = sparse_x[indices]
            sparse_y = sparse_y[indices]

        labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    elif label_emb == 'glove':
        emb_init = get_word_emb(emb_init)
        train_x, train_y = get_data(train_x, sparse_data_y)

        with redirect_stderr(None):
            train_y = mlb.transform(train_y)

        if indices is not None:
            train_x = train_x[indices]
            train_y = train_y[indices]

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


def neo_kmeans():
    pass


def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans'):
    if assign_labels not in ('kmeans', 'neo-kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given"
                         % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    # The first eigen vector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False)

    if assign_labels == 'kmeans':
        _, labels, _ = k_means(maps, n_clusters, random_state=random_state,
                               n_init=n_init)
    elif assign_labels == 'neo-kmeans':
        pass
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


class MySpectralClustering(SpectralClustering):
    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)

        y : Ignored

        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self
