import mlflow

from logzero import logger
from deepxml.data_utils import get_data


def load_dataset(data_cnf):
    logger.info('Loading Training and Validation Set')
    train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
    return train_x, train_labels


def log_config(data_cnf_path, model_cnf_path, dry_run):
    logger.info('Finish Training')
    if not dry_run:
        mlflow.log_artifact(data_cnf_path, 'config')
        mlflow.log_artifact(model_cnf_path, 'config')


def log_results(score_path, label_path, dry_run):
    if not dry_run:
        mlflow.log_artifact(score_path, 'results')
        mlflow.log_artifact(label_path, 'results')
        with open('run_id.txt', 'w') as f:
            f.write(mlflow.active_run().info.run_id)
