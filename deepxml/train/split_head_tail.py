import numpy as np

from logzero import logger

from contextlib import redirect_stderr

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import (
    get_head_tail_labels, get_head_tail_samples, get_data, get_mlb, output_res,
)
from deepxml.networks import AttentionRNN
from deepxml.models import Model

from .utils import load_dataset, log_config, log_results


def splitting_head_tail_train(
    data_cnf, data_cnf_path, model_cnf, model_cnf_path,
    emb_init, model_path, tree_id, output_suffix, dry_run,
    split_ratio,
):
    train_x, train_labels = load_dataset(data_cnf)

    logger.info(f'Split head and tail labels: {split_ratio}')
    head_labels, head_labels_i, tail_labels, tail_labels_i = get_head_tail_labels(
        train_labels,
        split_ratio,
    )

    train_h_x = train_x[head_labels_i]
    train_h_labels = train_labels[head_labels_i]

    train_t_x = train_x[tail_labels_i]
    train_t_labels = train_labels[tail_labels_i]

    if 'size' in data_cnf['valid']:
        valid_size = data_cnf['valid']['size']
        train_h_x, valid_h_x, train_h_labels, valid_h_labels = train_test_split(
            train_h_x, train_h_labels,
            test_size=valid_size if len(train_h_x) > 2 * valid_size else 0.1,
        )

        train_t_x, valid_t_x, train_t_labels, valid_t_labels = train_test_split(
            train_t_x, train_t_labels,
            test_size=valid_size if len(train_t_x) > 2 * valid_size else 0.1,
        )

    else:
        valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        valid_h_labels_i, valid_t_labels_i = get_head_tail_samples(
            head_labels, tail_labels, valid_labels,
        )
        valid_t_x = valid_x[valid_h_labels_i]
        valid_h_x = valid_x[valid_t_labels_i]
        valid_h_labels = valid_x[valid_h_labels_i]
        valid_t_labels = valid_x[valid_t_labels_i]

    labels_binarizer_path = data_cnf['labels_binarizer']
    mlb_h = get_mlb(f"{labels_binarizer_path}_h_{split_ratio}", head_labels[None, ...])
    mlb_t = get_mlb(f"{labels_binarizer_path}_t_{split_ratio}", tail_labels[None, ...])

    with redirect_stderr(None):
        train_h_y = mlb_h.transform(train_h_labels)
        valid_h_y = mlb_h.transform(valid_h_labels)
        train_t_y = mlb_t.transform(train_t_labels)
        valid_t_y = mlb_t.transform(valid_t_labels)

    logger.info(f'Number of Head Labels: {len(head_labels):,}')
    logger.info(f'Number of Tail Labels: {len(tail_labels):,}')
    logger.info(f'Size of Head Training Set: {len(train_h_x):,}')
    logger.info(f'Size of Head Validation Set: {len(valid_h_x):,}')
    logger.info(f'Size of Tail Training Set: {len(train_t_x):,}')
    logger.info(f'Size of Tail Validation Set: {len(valid_t_x):,}')

    logger.info('Training')
    if 'cluster' not in model_cnf:
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
        raise Exception("FastAttention is not currently supported for "
                        "splited head and tail dataset")

    log_config(data_cnf_path, model_cnf_path, dry_run)

    return head_model, tail_model, head_labels, tail_labels


def splitting_head_tail_eval(
    data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
    tree_id, output_suffix, dry_run, split_ratio, head_labels, tail_labels,
    head_model, tail_model,
):
    logger.info('Loading Test Set')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x):,}')

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

        if head_model is None:
            head_model = Model(
                network=AttentionRNN, labels_num=len(head_labels),
                model_path=f'{model_path}-head', emb_init=emb_init,
                load_model=True,
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
                load_model=True,
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
        labels = labels[i, j]
    else:
        raise Exception("FastAttention is not currently supported for "
                        "splited head and tail dataset")

    logger.info('Finish Predicting')
    score_path, label_path = output_res(data_cnf['output']['res'],
                                        f'{model_name}-{data_name}{tree_id}',
                                        scores, labels, output_suffix)

    log_results(score_path, label_path, dry_run)
