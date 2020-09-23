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

import click

import numpy as np
import torch.distributed as dist

from logzero import logger
from ruamel.yaml import YAML

from deepxml.data_utils import get_word_emb

from deepxml.train import (
    default_train, default_eval, splitting_head_tail_train, splitting_head_tail_eval,
    random_forest_train, random_forest_eval, spectral_clustering_train,
    spectral_clustering_eval,
)


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
    is_random_forest = 'random_forest' in model_cnf
    is_spectral_clustering = 'spectral_clustering' in model_cnf

    if is_split_head_tail:
        split_ratio = data_cnf['split_head_tail']
        head_model = None
        tail_model = None
        head_labels = None
        tail_labels = None

    elif is_random_forest:
        num_tree = model_cnf['random_forest']['num']

    elif is_spectral_clustering:
        pass

    if mode is None or mode == 'train':
        if is_split_head_tail:
            head_model, tail_model, head_labels, tail_labels = splitting_head_tail_train(
                data_cnf, data_cnf_path, model_cnf, model_cnf_path, emb_init,
                model_path, tree_id, output_suffix, dry_run, split_ratio,

            )

        elif is_random_forest:
            random_forest_train(
                data_cnf, data_cnf_path, model_cnf, model_cnf_path, emb_init,
                model_path, tree_id, output_suffix, dry_run, num_tree,
            )

        elif is_spectral_clustering:
            spectral_clustering_train(
                data_cnf, data_cnf_path, model_cnf, model_cnf_path, emb_init,
                model_path, tree_id, output_suffix, dry_run,
            )

        else:
            default_train(
                data_cnf, data_cnf_path, model_cnf, model_cnf_path, emb_init,
                model_path, tree_id, output_suffix, dry_run,
            )

        log_tag(dry_run, model_name, data_name, output_suffix)

    if mode is None or mode == 'eval':
        if is_split_head_tail:
            splitting_head_tail_eval(
                data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
                tree_id, output_suffix, dry_run, split_ratio, head_labels, tail_labels,
                head_model, tail_model,
            )

        elif is_random_forest:
            random_forest_eval(
                data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
                tree_id, output_suffix, dry_run, num_tree,
            )

        elif is_spectral_clustering:
            spectral_clustering_eval(
                data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
                tree_id, output_suffix, dry_run,
            )

        else:
            default_train(
                data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
                tree_id, output_suffix, dry_run,
            )


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
