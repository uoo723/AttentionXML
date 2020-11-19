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

    input_ids = []
    attention_mask = []

    for line in tqdm(lines, desc='Tokenized...'):
        inputs = tokenizer(line, truncation=True, padding="max_length", return_tensors="np")
        input_ids.append(inputs['input_ids'])
        attention_mask.append(inputs['attention_mask'])

    input_ids = np.concatenate(input_ids)
    attention_mask = np.concatenate(attention_mask)

    logger.info("Save tokenized input")
    np.savez(tokenized_path, input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    main()
