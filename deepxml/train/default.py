import numpy as np

from logzero import logger
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, output_res
from deepxml.networks import AttentionRNN
from deepxml.models import Model
from deepxml.tree import FastAttentionXML
from deepxml.evaluation import get_inv_propensity

from .utils import load_dataset, log_config, log_results


def default_train(
    data_cnf, data_cnf_path, model_cnf, model_cnf_path,
    emb_init, model_path, tree_id, output_suffix, dry_run,
):
    train_x, train_labels = load_dataset(data_cnf)

    if 'size' in data_cnf['valid']:
        train_x, valid_x, train_labels, valid_labels = train_test_split(
            train_x, train_labels, test_size=data_cnf['valid']['size'],
        )

    else:
        valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])

    mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((
        train_labels, valid_labels,
    )))
    freq = mlb.transform(np.hstack([train_labels, valid_labels])).sum(axis=0).A1
    train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
    labels_num = len(mlb.classes_)
    logger.info(F'Number of Labels: {labels_num}')
    logger.info(F'Size of Training Set: {len(train_x):,}')
    logger.info(F'Size of Validation Set: {len(valid_x):,}')

    logger.info('Training')
    if 'cluster' not in model_cnf:
        if 'propensity' in data_cnf:
            a = data_cnf['propensity']['a']
            b = data_cnf['propensity']['b']
            pos_weight = get_inv_propensity(train_y, a, b)
        else:
            pos_weight = None

        train_loader = DataLoader(
            MultiLabelDataset(train_x, train_y),
            model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
        valid_loader = DataLoader(
            MultiLabelDataset(valid_x, valid_y, training=False),
            model_cnf['valid']['batch_size'], num_workers=4)

        if 'loss' in model_cnf:
            gamma = model_cnf['loss'].get('gamma', 2.0)
            loss_name = model_cnf['loss']['name']
        else:
            gamma = None
            loss_name = 'bce'

        model = Model(
            network=AttentionRNN, labels_num=labels_num, model_path=model_path,
            emb_init=emb_init, pos_weight=pos_weight, loss_name=loss_name, gamma=gamma,
            freq=freq, **data_cnf['model'], **model_cnf['model'])

        if not dry_run:
            model.train(train_loader, valid_loader, mlb=mlb, **model_cnf['train'])
        else:
            model.save_model()

    else:
        model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id, output_suffix)

        if not dry_run:
            model.train(train_x, train_y, valid_x, valid_y, mlb)

    log_config(data_cnf_path, model_cnf_path, dry_run)


def default_eval(
    data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
    tree_id, output_suffix, dry_run,
):
    logger.info('Loading Test Set')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x):,}')

    logger.info('Predicting')
    model_cnf['model'].pop('load_model', None)
    if 'cluster' not in model_cnf:
        test_loader = DataLoader(
            MultiLabelDataset(test_x),
            model_cnf['predict']['batch_size'],
            num_workers=4)

        if 'loss' in model_cnf:
            gamma = model_cnf['loss'].get('gamma', 1.0)
            loss_name = model_cnf['loss']['name']
        else:
            gamma = None
            loss_name = 'bce'

        model = Model(
            network=AttentionRNN, labels_num=labels_num,
            model_path=model_path, emb_init=emb_init,
            load_model=True, loss_name=loss_name, gamma=gamma,
            **data_cnf['model'], **model_cnf['model'])

        scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
        labels = mlb.classes_[labels]
    else:
        model = FastAttentionXML(labels_num, data_cnf, model_cnf,
                                 tree_id, output_suffix)

        scores, labels = model.predict(test_x, model_cnf['predict'].get('k', 100))
        labels = mlb.classes_[labels]

    logger.info('Finish Predicting')
    score_path, label_path = output_res(data_cnf['output']['res'],
                                        f'{model_name}-{data_name}{tree_id}',
                                        scores, labels, output_suffix)

    log_results(score_path, label_path, dry_run)
