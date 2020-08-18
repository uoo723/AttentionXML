#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2019/8/21
@author yrh

"""

import warnings
warnings.filterwarnings('ignore')

import click
import mlflow

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from deepxml.evaluation import *


@click.command()
@click.option('-r', '--results', type=click.Path(exists=True), help='Path of results.')
@click.option('-t', '--targets', type=click.Path(exists=True), help='Path of targets.')
@click.option('--train-labels', type=click.Path(exists=True), default=None, help='Path of labels for training set.')
@click.option('-a', type=click.FLOAT, default=0.55, help='Parameter A for propensity score.')
@click.option('-b', type=click.FLOAT, default=1.5, help='Parameter B for propensity score.')
def main(results, targets, train_labels, a, b):
    res, targets = np.load(results, allow_pickle=True), np.load(targets, allow_pickle=True)
    mlb = MultiLabelBinarizer(sparse_output=True)
    targets = mlb.fit_transform(targets)

    p1 = get_p_1(res, targets, mlb)
    p3 = get_p_3(res, targets, mlb)
    p5 = get_p_5(res, targets, mlb)

    mlflow.log_metric("p1", p1)
    mlflow.log_metric("p3", p3)
    mlflow.log_metric("p5", p5)

    print('Precision@1,3,5:', p1, p3, p5)

    n1 = get_n_1(res, targets, mlb)
    n3 = get_n_3(res, targets, mlb)
    n5 = get_n_5(res, targets, mlb)

    mlflow.log_metric("n1", n1)
    mlflow.log_metric("n3", n3)
    mlflow.log_metric("n5", n5)

    print('nDCG@1,3,5:', n1, n3, n5)

    r1 = get_r_1(res, targets, mlb)
    r3 = get_r_3(res, targets, mlb)
    r5 = get_r_5(res, targets, mlb)

    print('Recall@1,3,5:', r1, r3, r5)

    if train_labels is not None:
        train_labels = np.load(train_labels, allow_pickle=True)
        inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)

        psp1 = get_psp_1(res, targets, inv_w, mlb)
        psp3 = get_psp_3(res, targets, inv_w, mlb)
        psp5 = get_psp_5(res, targets, inv_w, mlb)

        mlflow.log_metric("psp1", psp1)
        mlflow.log_metric("psp3", psp3)
        mlflow.log_metric("psp5", psp5)

        print('PSPrecision@1,3,5:', psp1, psp3, psp5)

        psndcg1 = get_psndcg_1(res, targets, inv_w, mlb)
        psndcg3 = get_psndcg_3(res, targets, inv_w, mlb)
        psndcg5 = get_psndcg_5(res, targets, inv_w, mlb)

        mlflow.log_metric("psndcg1", psndcg1)
        mlflow.log_metric("psndcg3", psndcg3)
        mlflow.log_metric("psndcg5", psndcg5)

        print('PSnDCG@1,3,5:', psndcg1, psndcg3, psndcg5)


if __name__ == '__main__':
    main()
