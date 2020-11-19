#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/10
@author yrh

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Sequence, Optional, Union


__all__ = ['MultiLabelDataset', 'XMLDataset']

TDataX = Sequence[Sequence]
TDataY = Optional[csr_matrix]
TCandidate = TGroup = Optional[np.ndarray]
TGroupLabel = TGroupScore = Optional[Union[np.ndarray, torch.Tensor]]


class MultiLabelDataset(Dataset):
    """

    """
    def __init__(self, data_x: TDataX, data_y: TDataY = None,
                 attention_mask: TDataX = None, training=True):
        self.data_x, self.data_y, self.training = data_x, data_y, training
        self.attention_mask = attention_mask

    def __getitem__(self, item):
        ret = self.data_x[item]
        if self.attention_mask is not None:
            ret = (ret, self.attention_mask[item])

        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
            ret = (*ret, data_y) if isinstance(ret, tuple) else (ret, data_y)

        return ret

    def __len__(self):
        return len(self.data_x)


class XMLDataset(MultiLabelDataset):
    """

    """
    def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True,
                 labels_num=None, candidates: TCandidate = None, candidates_num=None,
                 groups: TGroup = None, group_labels: TGroupLabel = None, group_scores: TGroupScore = None):
        super(XMLDataset, self).__init__(data_x, data_y, training)
        self.labels_num, self.candidates, self.candidates_num = labels_num, candidates, candidates_num
        self.groups, self.group_labels, self.group_scores = groups, group_labels, group_scores

        is_overlapping = False

        if self.candidates is None:
            self.candidates = [np.concatenate([self.groups[g] for g in group_labels])
                               for group_labels in tqdm(self.group_labels, leave=False, desc='Candidates')]

            if len(self.candidates[0]) != len(np.unique(self.candidates[0])):
                is_overlapping = True

            if self.group_scores is not None:
                self.candidates_scores = [np.concatenate([[s] * len(self.groups[g])
                                                          for g, s in zip(group_labels, group_scores)])
                                          for group_labels, group_scores in zip(self.group_labels, self.group_scores)]

                if is_overlapping:
                    candidates_scores = []
                    candidates, inverses = zip(
                        *map(lambda x: np.unique(x, return_inverse=True), self.candidates))

                    for i, inverse in tqdm(enumerate(inverses), leave=False, desc="Scores",
                                           total=len(inverses)):
                        n = len(candidates[i])
                        rows, cols = np.where(inverse == np.arange(n)[:, None])
                        _, inverse_rows = np.unique(rows, return_index=True)
                        res = np.split(cols, inverse_rows[1:])

                        s = self.candidates_scores[i]
                        candidates_scores.append(np.array([np.max(s[i]) for i in res]))

                    self.candidates = list(candidates)
                    self.candidates_scores = candidates_scores

            elif is_overlapping:
                self.candidates = list(map(lambda x: np.unique(x), self.candidates))

        else:
            self.candidates_scores = [np.ones_like(candidates) for candidates in self.candidates]
        if self.candidates_num is None:
            self.candidates_num = self.group_labels.shape[1] * max(len(g) for g in groups)

    def __getitem__(self, item):
        data_x, candidates = self.data_x[item], np.asarray(self.candidates[item], dtype=np.int)
        if self.training and self.data_y is not None:
            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.labels_num, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)
            data_y = self.data_y[item, candidates].toarray().squeeze(0).astype(np.float32)
            return (data_x, candidates), data_y
        else:
            scores = self.candidates_scores[item]
            if len(candidates) < self.candidates_num:
                scores = np.concatenate([scores, [-np.inf] * (self.candidates_num - len(candidates))])
                candidates = np.concatenate([candidates, [self.labels_num] * (self.candidates_num - len(candidates))])
            scores = np.asarray(scores, dtype=np.float32)
            return data_x, candidates, scores

    def __len__(self):
        return len(self.data_x)
