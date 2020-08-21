#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import os
import random
import torch

from pathlib import Path
from contextlib import redirect_stderr

import click

import mlflow
import mlflow.pytorch

import numpy as np
import torch.distributed as dist

from logzero import logger
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from deepxml.data_utils import (
    get_data, get_mlb, get_word_emb, output_res, get_head_tail_labels,
    get_head_tail_samples,
)

from deepxml.dataset import MultiLabelDataset
from deepxml.models import Model
from deepxml.networks import AttentionRNN
from deepxml.tree import FastAttentionXML


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('-t', '--tree-id', type=click.INT, default=None)
@click.option('-s', '--output-suffix', type=click.STRING, default='', help='suffix of output name')
@click.option('--dry-run', is_flag=True, default=False, help='dry run for test code')
def main(data_cnf, model_cnf, mode, tree_id, output_suffix, dry_run):
    set_seed(tree_id)

    tree_id = F'-Tree-{tree_id}' if tree_id is not None else ''
    yaml = YAML(typ='safe')

    data_cnf_path = data_cnf
    model_cnf_path = model_cnf

    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}{output_suffix}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    is_split_head_tail = 'split_head_tail' in data_cnf

    if is_split_head_tail:
        head_model = None
        tail_model = None
        head_labels = None
        tail_labels = None
        split_ratio = data_cnf['split_head_tail']

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])

        if is_split_head_tail:
            logger.info('Split head and tail labels')
            head_labels, head_labels_i, tail_labels, tail_labels_i = get_head_tail_labels(
                train_labels,
                split_ratio,
            )

            train_h_x = train_x[head_labels_i]
            train_h_labels = train_labels[head_labels_i]

            train_t_x = train_x[tail_labels_i]
            train_t_labels = train_labels[tail_labels_i]

        if 'size' in data_cnf['valid']:
            if is_split_head_tail:
                train_h_x, valid_h_x, train_h_labels, valid_h_labels = train_test_split(
                    train_h_x, train_h_labels, test_size=data_cnf['valid']['size'],
                )

                train_t_x, valid_t_x, train_t_labels, valid_t_labels = train_test_split(
                    train_t_x, train_t_labels, test_size=data_cnf['valid']['size'],
                )

            else:
                train_x, valid_x, train_labels, valid_labels = train_test_split(
                    train_x, train_labels, test_size=data_cnf['valid']['size'],
                )

        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])

            if is_split_head_tail:
                valid_h_labels_i, valid_t_labels_i = get_head_tail_samples(
                    head_labels, tail_labels, valid_labels,
                )
                valid_t_x = valid_x[valid_h_labels_i]
                valid_h_x = valid_x[valid_t_labels_i]
                valid_h_labels = valid_x[valid_h_labels_i]
                valid_t_labels = valid_x[valid_t_labels_i]

        if is_split_head_tail:
            labels_binarizer_path = data_cnf['labels_binarizer']
            mlb_h = get_mlb(f"{labels_binarizer_path}_h_{split_ratio}", head_labels[None, ...])
            mlb_t = get_mlb(f"{labels_binarizer_path}_t_{split_ratio}", tail_labels[None, ...])

            with redirect_stderr(None):
                train_h_y = mlb_h.transform(train_h_labels)
                valid_h_y = mlb_h.transform(valid_h_labels)
                train_t_y = mlb_t.transform(train_t_labels)
                valid_t_y = mlb_t.transform(valid_t_labels)

            logger.info(f'Number of Head Labels: {len(head_labels)}')
            logger.info(f'Number of Tail Labels: {len(tail_labels)}')
            logger.info(f'Size of Head Training Set: {len(train_h_x)}')
            logger.info(f'Size of Head Validation Set: {len(valid_h_x)}')
            logger.info(f'Size of Tail Training Set: {len(train_t_x)}')
            logger.info(f'Size of Tail Validation Set: {len(valid_t_x)}')
        else:
            mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((
                train_labels, valid_labels,
            )))
            train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
            labels_num = len(mlb.classes_)
            logger.info(F'Number of Labels: {labels_num}')
            logger.info(F'Size of Training Set: {len(train_x)}')
            logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        if 'cluster' not in model_cnf:
            if is_split_head_tail:
                train_h_loader = DataLoader(
                    MultiLabelDataset(train_h_x, train_h_y),
                    model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
                valid_h_loader = DataLoader(
                    MultiLabelDataset(valid_h_x, valid_h_y, training=False),
                    model_cnf['valid']['batch_size'], num_workers=4)
                head_model = Model(
                    network=AttentionRNN, labels_num=len(head_labels),
                    model_path=f'{model_path}-head', emb_init=emb_init,
                    **data_cnf['model'], **model_cnf['model'])

                if not dry_run:
                    logger.info('Training Head Model')
                    head_model.train(train_h_loader, valid_h_loader, **model_cnf['train'])
                    logger.info('Finish Traning Head Model')
                else:
                    head_model.save_model()

                train_t_loader = DataLoader(
                    MultiLabelDataset(train_t_x, train_t_y),
                    model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
                valid_t_loader = DataLoader(
                    MultiLabelDataset(valid_t_x, valid_t_y, training=False),
                    model_cnf['valid']['batch_size'], num_workers=4)
                tail_model = Model(
                    network=AttentionRNN, labels_num=len(tail_labels),
                    model_path=f'{model_path}-tail', emb_init=emb_init,
                    **data_cnf['model'], **model_cnf['model'])

                if not dry_run:
                    logger.info('Training Tail Model')
                    tail_model.train(train_t_loader, valid_t_loader, **model_cnf['train'])
                    logger.info('Finish Traning Tail Model')
                else:
                    tail_model.save_model()

            else:
                train_loader = DataLoader(
                    MultiLabelDataset(train_x, train_y),
                    model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
                valid_loader = DataLoader(
                    MultiLabelDataset(valid_x, valid_y, training=False),
                    model_cnf['valid']['batch_size'], num_workers=4)
                model = Model(
                    network=AttentionRNN, labels_num=labels_num, model_path=model_path,
                    emb_init=emb_init, **data_cnf['model'], **model_cnf['model'])

                # mlflow.pytorch.log_model(model.model.module, 'model')
                if not dry_run:
                    model.train(train_loader, valid_loader, **model_cnf['train'])
                else:
                    model.save_model()
        else:
            if is_split_head_tail:
                raise Exception("FastAttention is not currently supported for "
                                "splited head and tail dataset")
            model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id, output_suffix)

            if not dry_run:
                model.train(train_x, train_y, valid_x, valid_y, mlb)
        logger.info('Finish Training')

        if not dry_run:
            mlflow.log_artifact(data_cnf_path, 'config')
            mlflow.log_artifact(model_cnf_path, 'config')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        if is_split_head_tail:
            labels_binarizer_path = data_cnf['labels_binarizer']
            mlb_h = get_mlb(f"{labels_binarizer_path}_h_{split_ratio}")
            mlb_t = get_mlb(f"{labels_binarizer_path}_t_{split_ratio}")

            if head_labels is None:
                train_x, train_labels = get_data(data_cnf['train']['texts'],
                                                 data_cnf['train']['labels'])
                head_labels, _, tail_labels, _ = get_head_tail_labels(
                    train_labels,
                    split_ratio,
                )

            h_labels_i = np.nonzero(mlb.transform(head_labels[None, ...]).toarray())[0]
            t_labels_i = np.nonzero(mlb.transform(tail_labels[None, ...]).toarray())[0]

        logger.info('Predicting')
        if 'cluster' not in model_cnf:
            test_loader = DataLoader(
                MultiLabelDataset(test_x),
                model_cnf['predict']['batch_size'],
                num_workers=4)

            if is_split_head_tail:
                if head_model is None:
                    head_model = Model(
                        network=AttentionRNN, labels_num=len(head_labels),
                        model_path=f'{model_path}-head', emb_init=emb_init,
                        **data_cnf['model'], **model_cnf['model'])

                logger.info('Predicting Head Model')
                h_k = model_cnf['predict'].get('top_head_k', 30)
                scores_h, labels_h = head_model.predict(test_loader, k=h_k)
                labels_h = mlb_h.classes_[labels_h]
                logger.info('Finish Predicting Head Model')

                if tail_model is None:
                    tail_model = Model(
                        network=AttentionRNN, labels_num=len(tail_labels),
                        model_path=f'{model_path}-tail', emb_init=emb_init,
                        **data_cnf['model'], **model_cnf['model'])

                logger.info('Predicting Tail Model')
                t_k = model_cnf['predict'].get('top_tail_k', 70)
                scores_t, labels_t = tail_model.predict(test_loader, k=t_k)
                labels_t = mlb_t.classes_[labels_t]
                logger.info('Finish Predicting Tail Model')

                scores = np.c_[scores_h, scores_t]
                labels = np.c_[labels_h, labels_t]

                i = np.arange(len(scores))[:, None]
                j = np.argsort(scores)[:, ::-1]

                scores = scores[i, j]
                labels =labels[i, j]

            else:
                if model is None:
                    model = Model(
                        network=AttentionRNN, labels_num=labels_num,
                        model_path=model_path, emb_init=emb_init,
                        **data_cnf['model'], **model_cnf['model'])

                scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
                labels = mlb.classes_[labels]
        else:
            if is_split_head_tail:
                raise Exception("FastAttention is not currently supported for "
                                "splited head and tail dataset")
            if model is None:
                model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id, output_suffix)
            scores, labels = model.predict(test_x)
            labels = mlb.classes_[labels]

        logger.info('Finish Predicting')
        score_path, label_path = output_res(data_cnf['output']['res'],
                                            f'{model_name}-{data_name}{tree_id}',
                                            scores, labels, output_suffix)

        if mode is None and not dry_run:
            mlflow.log_artifact(score_path, 'results')
            mlflow.log_artifact(label_path, 'results')

    if not dry_run:
        with open('run_id.txt', 'w') as f:
            f.write(mlflow.active_run().info.run_id)


def distributed_train(gpu, args):
    """GPU Distributed training"""
    rank = args.nr
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank,
    )


if __name__ == '__main__':
    main()
