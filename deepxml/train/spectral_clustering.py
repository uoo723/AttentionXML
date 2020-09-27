import numpy as np

from contextlib import redirect_stderr
from logzero import logger

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, output_res, get_splitted_samples
from deepxml.networks import AttentionRNN
from deepxml.models import Model
from deepxml.tree import FastAttentionXML
from deepxml.cluster import MySpectralClustering

from .utils import load_dataset, log_config, log_results


def spectral_clustering_train(
    data_cnf, data_cnf_path, model_cnf, model_cnf_path,
    emb_init, model_path, tree_id, output_suffix, dry_run,
):
    train_xs = []
    valid_xs = []
    train_labels_list = []
    valid_labels_list = []
    train_ys = []
    valid_ys = []
    mlb_list = []
    indices_list = []

    n_clusters = model_cnf['spectral_clustering']['num_clusters']
    n_components = model_cnf['spectral_clustering']['n_components']
    alg = model_cnf['spectral_clustering']['alg']
    size_min = model_cnf['spectral_clustering']['size_min']
    size_max = model_cnf['spectral_clustering']['size_max']

    train_x, train_labels = load_dataset(data_cnf)

    if 'cluster' not in model_cnf:
        mlb = get_mlb(data_cnf['labels_binarizer'], train_labels)
        train_y = mlb.transform(train_labels)

        logger.info('Build label adjacency matrix')
        adj = train_y.T @ train_y
        adj.setdiag(0)
        adj.eliminate_zeros()
        logger.info(f"Sparsity: {adj.count_nonzero() / adj.shape[0] ** 2}")
        clustering = MySpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                          n_components=n_components, n_init=1,
                                          size_min=size_min,
                                          size_max=size_max,
                                          assign_labels=alg, n_jobs=-1)
        logger.info('Start Spectral Clustering')
        clustering.fit(adj)
        logger.info('Finish Spectral Clustering')

        groups = [[] for _ in range(n_clusters)]
        for i, group in enumerate(clustering.labels_):
            groups[group].append(i)

        splitted_labels = []
        for indices in groups:
            splitted_labels.append(mlb.classes_[indices])

        for labels in splitted_labels:
            indices = get_splitted_samples(labels, train_labels)
            indices_list.append(indices)
            train_xs.append(train_x[indices])
            train_labels_list.append(train_labels[indices])

        if 'size' in data_cnf['valid']:
            for i, (train_x, train_labels) in enumerate(zip(train_xs, train_labels_list)):
                valid_size = data_cnf['valid']['size']
                if len(train_x) * 0.8 > len(train_x) - valid_size:
                    valid_size = 0.2
                train_x, valid_x, train_labels, valid_labels = train_test_split(
                    train_x, train_labels, test_size=valid_size,
                )
                train_xs[i] = train_x
                train_labels_list[i] = train_labels
                valid_xs.append(valid_x)
                valid_labels_list.append(valid_labels)

        else:
            raise Exception("Setting valid set explicitly is not "
                            "supported spectral clustering mode.")

        labels_binarizer_path = data_cnf['labels_binarizer']
        suffix = output_suffix.upper().replace('-', '_')
        for i, labels in enumerate(splitted_labels):
            filename = f"{labels_binarizer_path}_{suffix}_{i}"
            mlb_tree = get_mlb(filename, labels[None, ...], force=True)
            mlb_list.append(mlb_tree)
            logger.info(f"Number of labels of cluster {i}: {len(labels):,}")
            logger.info(f"Number of Training Set of cluster {i}: {len(train_xs[i]):,}")
            logger.info(f"Number of Validation Set of cluster {i}: {len(valid_xs[i]):,}")

            with redirect_stderr(None):
                train_y = mlb_tree.transform(train_labels_list[i])
                valid_y = mlb_tree.transform(valid_labels_list[i])

            train_ys.append(train_y)
            valid_ys.append(valid_y)

    else:
        if 'size' in data_cnf['valid']:
            train_x, valid_x, train_labels, valid_labels = train_test_split(
                train_x, train_labels, test_size=data_cnf['valid']['size'],
            )

        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])

        mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((
            train_labels, valid_labels,
        )))

        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)


    logger.info('Training')
    if 'cluster' not in model_cnf:
        for i, (train_x, train_y, valid_x, valid_y) in enumerate(zip(
            train_xs, train_ys, valid_xs, valid_ys,
        )):
            train_loader = DataLoader(
                MultiLabelDataset(train_x, train_y),
                model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
            valid_loader = DataLoader(
                MultiLabelDataset(valid_x, valid_y, training=False),
                model_cnf['valid']['batch_size'], num_workers=4)
            model = Model(
                network=AttentionRNN, labels_num=len(mlb_list[i].classes_),
                model_path=f'{model_path}-{i}', emb_init=emb_init,
                **data_cnf['model'], **model_cnf['model'])

            if not dry_run:
                logger.info(f"Start Training Cluster {i}")
                model.train(train_loader, valid_loader, **model_cnf['train'])
                logger.info(f"Finish Training Cluster {i}")
            else:
                model.save_model()

    else:
        model = FastAttentionXML(
            len(mlb.classes_), data_cnf, model_cnf, tree_id, output_suffix,
        )

        if not dry_run:
            model.train(train_x, train_y, valid_x, valid_y, mlb)

    log_config(data_cnf_path, model_cnf_path, dry_run)


def spectral_clustering_eval(
    data_cnf, model_cnf, data_name, model_name, model_path, emb_init,
    tree_id, output_suffix, dry_run,
):
    mlb_list = []
    n_clusters = model_cnf['spectral_clustering']['num_clusters']
    labels_binarizer_path = data_cnf['labels_binarizer']
    scores_list = []
    labels_list = []

    logger.info('Loading Test Set')
    test_x, _ = get_data(data_cnf['test']['texts'], None)
    logger.info(F'Size of Test Set: {len(test_x):,}')

    logger.info('Predicting')
    if 'cluster' not in model_cnf:
        suffix = output_suffix.upper().replace('-', '_')
        for i in range(n_clusters):
            filename = f"{labels_binarizer_path}_{suffix}_{i}"
            mlb_tree = get_mlb(filename)
            mlb_list.append(mlb_tree)

        test_loader = DataLoader(
            MultiLabelDataset(test_x),
            model_cnf['predict']['batch_size'],
            num_workers=4)

        for i, mlb in enumerate(mlb_list):
            logger.info(f"Predicting Cluster {i}")
            labels_num = len(mlb.classes_)
            k = model_cnf['predict'].get('k', 100) // n_clusters

            model = Model(
                network=AttentionRNN, labels_num=labels_num,
                model_path=f'{model_path}-{i}', emb_init=emb_init,
                load_model=True,
                **data_cnf['model'], **model_cnf['model'])

            scores, labels = model.predict(test_loader, k=k)
            scores_list.append(scores)
            labels_list.append(mlb.classes_[labels])
            logger.info(f"Finish Prediting Cluster {i}")

        scores = np.hstack(scores_list)
        labels = np.hstack(labels_list)

        i = np.arange(len(scores))[:, None]
        j = np.argsort(scores)[:, ::-1]

        scores = scores[i, j]
        labels = labels[i, j]

    else:
        mlb = get_mlb(data_cnf['labels_binarizer'])
        model = FastAttentionXML(len(mlb.classes_), data_cnf, model_cnf,
                                 tree_id, output_suffix)

        scores, labels = model.predict(test_x, model_cnf['predict'].get('k', 100))
        labels = mlb.classes_[labels]

    logger.info('Finish Predicting')
    score_path, label_path = output_res(data_cnf['output']['res'],
                                        f'{model_name}-{data_name}{tree_id}',
                                        scores, labels, output_suffix)

    log_results(score_path, label_path, dry_run)
