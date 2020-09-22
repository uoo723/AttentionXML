import numpy as np

from contextlib import redirect_stderr

from logzero import logger

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import (
    get_unique_labels, get_splitted_samples, split_labels, get_mlb, output_res,
    get_data,
)
from deepxml.tree import FastAttentionXML
from deepxml.networks import AttentionRNN
from deepxml.models import Model

from .utils import load_dataset, log_config, log_results


def random_forest_train(
    data_cnf, data_cnf_path, model_cnf, model_cnf_path,
    emb_init, model_path, tree_id, output_suffix, dry_run,
    num_tree,
):
    indices_list = []
    train_xs = []
    valid_xs = []
    train_labels_list = []
    valid_labels_list = []
    train_ys = []
    valid_ys = []
    mlb_list = []

    train_x, train_labels = load_dataset(data_cnf)

    unique_labels = get_unique_labels(train_labels)
    splitted_labels = split_labels(unique_labels, num_tree)

    for labels in splitted_labels:
        indices = get_splitted_samples(labels, train_labels)
        indices_list.append(indices)
        train_xs.append(train_x[indices])
        train_labels_list.append(train_labels[indices])

    if 'size' in data_cnf['valid']:
        valid_size = data_cnf['valid']['size']
        for i, (train_x, train_labels) in enumerate(zip(train_xs, train_labels_list)):
            train_x, valid_x, train_labels, valid_labels = train_test_split(
                train_x, train_labels, test_size=valid_size,
            )
            train_xs[i] = train_x
            train_labels_list[i] = train_labels
            valid_xs.append(valid_x)
            valid_labels_list.append(valid_labels)

    else:
        raise Exception("Setting valid set explicitly is not "
                        "supported random forest mode.")

    labels_binarizer_path = data_cnf['labels_binarizer']
    mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((
        train_labels, valid_labels,
    )))

    for i, labels in enumerate(splitted_labels):
        filename = f"{labels_binarizer_path}_RF_{i}"
        mlb_tree = get_mlb(filename, labels[None, ...])
        mlb_list.append(mlb_tree)
        logger.info(f"Number of labels of Tree {i}: {len(labels):,}")
        logger.info(f"Number of Training Set of Tree {i}: {len(train_xs[i]):,}")
        logger.info(f"Number of Validation Set of Tree {i}: {len(valid_xs[i]):,}")

        with redirect_stderr(None):
            train_y = mlb_tree.transform(train_labels_list[i])
            valid_y = mlb_tree.transform(valid_labels_list[i])

        train_ys.append(train_y)
        valid_ys.append(valid_y)

        logger.info('Training')
        if 'cluster' not in model_cnf:
            raise Exception("AttentionXML is not currently supported for "
                            "random forest mode")

        else:
            for i, (train_x, train_y, valid_x, valid_y, indices) in enumerate(zip(
                train_xs, train_ys, valid_xs, valid_ys, indices_list
            )):
                model = FastAttentionXML(
                    len(mlb_list[i].classes_), data_cnf, model_cnf, tree_id,
                    f"{output_suffix}-{i}")

                if not dry_run:
                    logger.info(f"Start Training RF {i}")
                    model.train(train_x, train_y, valid_x, valid_y, mlb_list[i], indices)
                    logger.info(f"Finish Training RF {i}")

        log_config(data_cnf_path, model_cnf_path, dry_run)


def random_forest_eval(
    data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
    tree_id, output_suffix, dry_run, num_tree,
):
    mlb_list = []
    logger.info('Loading Test Set')
    mlb = get_mlb(data_cnf['labels_binarizer'])
    labels_num = len(mlb.classes_)
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x):,}')

    logger.info('Predicting')
    if 'cluster' not in model_cnf:
        raise Exception("AttentionXML is not currently supported random forest mode")
    else:
        labels_binarizer_path = data_cnf['labels_binarizer']
        for i in range(num_tree):
            filename = f"{labels_binarizer_path}_RF_{i}"
            mlb_tree = get_mlb(filename)
            mlb_list.append(mlb_tree)

        scores_list = []
        labels_list = []

        for i, mlb in enumerate(mlb_list):
            logger.info(f"Predicting RF {i}")
            model = FastAttentionXML(
                        len(mlb.classes_), data_cnf, model_cnf, tree_id,
                        f"{output_suffix}-{i}")
            scores, labels = model.predict(test_x, model_cnf['predict'].get('rf_k', 100 // num_tree))
            scores_list.append(scores)
            labels_list.append(mlb.classes_[labels])
            logger.info(f"Finish Prediting RF {i}")

        scores = np.hstack(scores_list)
        labels = np.hstack(labels_list)

        i = np.arange(len(scores))[:, None]
        j = np.argsort(scores)[:, ::-1]

        scores = scores[i, j]
        labels = labels[i, j]

    logger.info('Finish Predicting')
    score_path, label_path = output_res(data_cnf['output']['res'],
                                        f'{model_name}-{data_name}{tree_id}',
                                        scores, labels, output_suffix)

    log_results(score_path, label_path, dry_run)
