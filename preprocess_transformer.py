#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/19
@author uoo723

"""

import os

import click
import numpy as np
from logzero import logger
from tqdm.auto import tqdm
from transformers import RobertaTokenizer


@click.command()
@click.option(
    "--text-path", type=click.Path(exists=True), required=True, help="Path of text."
)
@click.option(
    "--tokenized-path", type=click.Path(), required=True, help="Path of tokenized text."
)
@click.option(
    "--model-name",
    type=click.Choice(["roberta-base"]),
    default=None,
    help="Name of Pretrained model.",
)
def main(text_path, tokenized_path, model_name):
    if os.path.exists(tokenized_path):
        return

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    with open(text_path, "r") as fp:
        lines = fp.readlines()

    tokenized = np.concatenate(
        [
            tokenizer(line, truncation=True, padding="max_length", return_tensors="np")[
                "input_ids"
            ]
            for line in tqdm(lines, desc="tokenized...")
        ]
    )

    logger.info("Save tokenized input")
    np.save(tokenized_path, tokenized)


if __name__ == "__main__":
    main()
