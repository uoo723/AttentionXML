"""
Created on 2020/11/19
@author uoo723

"""
import numpy as np
from deepxml.data_utils import get_data, get_mlb, output_res
from deepxml.dataset import MultiLabelDataset
from deepxml.models import TransformerXML
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification

from .utils import load_dataset, log_config, log_results

MODEL_TYPE = {"roberta": RobertaForSequenceClassification}


def transformer_train(
    data_cnf, data_cnf_path, model_cnf, model_cnf_path, model_path, dry_run,
):
    train_x, train_labels = load_dataset(data_cnf)
    train_input_ids = train_x["input_ids"]
    train_atten_mask = train_x["attention_mask"]

    if "size" in data_cnf["valid"]:
        (
            train_x,
            valid_x,
            train_atten_mask,
            valid_atten_mask,
            train_labels,
            valid_labels,
        ) = train_test_split(
            train_input_ids,
            train_atten_mask,
            train_labels,
            test_size=data_cnf["valid"]["size"],
        )
    else:
        valid_x, valid_labels = get_data(
            data_cnf["valid"]["texts"], data_cnf["valid"]["labels"]
        )
        valid_atten_mask = None

    mlb = get_mlb(
        data_cnf["labels_binarizer"], np.hstack((train_labels, valid_labels,))
    )

    train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
    num_labels = len(mlb.classes_)
    logger.info(f"Number of Labels: {num_labels}")
    logger.info(f"Size of Training Set: {len(train_x):,}")
    logger.info(f"Size of Validation Set: {len(valid_x):,}")

    logger.info("Training")

    train_loader = DataLoader(
        MultiLabelDataset(train_x, train_y, train_atten_mask),
        model_cnf["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        MultiLabelDataset(valid_x, valid_y, valid_atten_mask, training=False),
        model_cnf["valid"]["batch_size"],
        num_workers=4,
    )

    model_cls = MODEL_TYPE[model_cnf["model"]["base"]]

    network = model_cls.from_pretrained(
        model_cnf["model"]["pretrained"], num_labels=num_labels
    )

    if model_cnf['model'].get('freeze_encoder', False):
        for param in network.base_model.parameters():
            param.requires_grad = False

    model = TransformerXML(
        network, model_path, **data_cnf["model"], **model_cnf["model"]
    )

    if not dry_run:
        model.train(train_loader, valid_loader, **model_cnf["train"])

    log_config(data_cnf_path, model_cnf_path, dry_run)


def transformer_eval(
    data_cnf,
    model_cnf,
    data_name,
    model_name,
    model_path,
    tree_id,
    output_suffix,
    dry_run,
):
    logger.info("Loading Test Set")
    mlb = get_mlb(data_cnf["labels_binarizer"])
    num_labels = len(mlb.classes_)
    test_x, _ = get_data(data_cnf["test"]["texts"], None)
    test_atten_mask = test_x["attention_mask"]
    test_x = test_x["input_ids"]

    logger.info(f"Size of Test Set: {len(test_x):,}")

    logger.info("Predicting")
    test_loader = DataLoader(
        MultiLabelDataset(test_x, attention_mask=test_atten_mask),
        model_cnf["predict"]["batch_size"],
        num_workers=4,
    )

    model_cls = MODEL_TYPE[model_cnf["model"]["base"]]

    network = model_cls.from_pretrained(
        model_cnf["model"]["pretrained"], num_labels=num_labels
    )

    model_cnf['model'].pop('load_model', None)
    model = TransformerXML(
        network, model_path, load_model=True, **data_cnf["model"], **model_cnf["model"]
    )

    scores, labels = model.predict(test_loader, k=model_cnf["predict"].get("k", 100))
    labels = mlb.classes_[labels]

    logger.info("Finish Predicting")
    score_path, label_path = output_res(
        data_cnf["output"]["res"],
        f"{model_name}-{data_name}{tree_id}",
        scores,
        labels,
        output_suffix,
    )

    log_results(score_path, label_path, dry_run)
