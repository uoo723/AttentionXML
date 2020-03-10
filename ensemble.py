#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2019/6/11
@author yrh

"""

import click
import numpy as np
from collections import defaultdict
from tqdm import tqdm


@click.command()
@click.option('-p', '--prefix', help='Prefix of results.')
@click.option('-i', '--input-suffix', default='', help='Suffix of input')
@click.option('-s', '--suffix', default='', help='Suffix of results')
@click.option('-t', '--trees', type=click.INT, help='The number of results using for ensemble.')
def main(prefix, input_suffix, suffix, trees):
    labels, scores = [], []
    for i in range(trees):
        labels.append(np.load(F'{prefix}-Tree-{i}-labels{input_suffix}.npy', allow_pickle=True))
        scores.append(np.load(F'{prefix}-Tree-{i}-scores{input_suffix}.npy', allow_pickle=True))
    ensemble_labels, ensemble_scores = [], []
    for i in tqdm(range(len(labels[0]))):
        s = defaultdict(float)
        for j in range(len(labels[0][i])):
            for k in range(trees):
                s[labels[k][i][j]] += scores[k][i][j]
        s = sorted(s.items(), key=lambda x: x[1], reverse=True)
        ensemble_labels.append([x[0] for x in s[:len(labels[0][i])]])
        ensemble_scores.append([x[1] for x in s[:len(labels[0][i])]])
    np.save(F'{prefix}-Ensemble-labels{suffix}', np.asarray(ensemble_labels))
    np.save(F'{prefix}-Ensemble-scores{suffix}', np.asarray(ensemble_scores))


if __name__ == '__main__':
    main()
