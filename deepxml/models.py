#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import contextlib
import os
from collections import deque
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from deepxml.data_utils import MixUp
from deepxml.evaluation import get_inv_propensity, get_n_5, get_p_5, get_psp_5
from deepxml.losses import *
from deepxml.modules import *
from deepxml.optimizers import *

# from apex import amp

__all__ = ['Model', 'XMLModel', 'TransformerXML']


class Model(object):
    """

    """
    def __init__(self, network, model_path, gradient_clip_value=5.0, device_ids=None,
                 load_model=False, pos_weight=None, loss_name='bce', gamma=2.0,
                 freq=None, mixup_opt=None, **kwargs):
        self.mixup_fn = None
        if mixup_opt is not None:
            logger.info('Mixup Enabled')
            logger.info(mixup_opt)
            self.mixup_opt = mixup_opt.copy()
            self.mixup_layer = mixup_opt.pop('layer', 'emb')
            self.mixup_warmup = mixup_opt.pop('warmup', 0)
            if self.mixup_layer not in ['emb', 'hidden']:
                raise ValueError(f'mixup_layer {self.mixup_layer} is invalid [`emb`, `hidden`]')
            self.mixup_fn = MixUp(**mixup_opt)

        if type(network) == type:
            self.model = nn.DataParallel(network(**kwargs).cuda(), device_ids=device_ids)
        else:
            self.model = nn.DataParallel(network.cuda(), device_ids=device_ids)

        # self.model = network(**kwargs).cuda()
        self.device_ids = device_ids

        if pos_weight is not None:
            logger.info('Inverse propensity weight enabled')
            pos_weight = torch.from_numpy(pos_weight).float().cuda()

        if loss_name == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_name == 'focal':
            self.loss_fn = FocalLoss(pos_weight=pos_weight, gamma=gamma)
        elif loss_name == 'bce+ranking':
            self.loss_fn = CombinedLoss(nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                                        RankingLoss(freq))
        elif loss_name == 'group_ranking':
            self.loss_fn = GroupRankingLoss()
        elif loss_name == 'bce+group':
            self.loss_fn = CombinedLoss(nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                                        GroupRankingLoss())
        else:
            raise ValueError(f"loss_name must be `bce` or `focal` or `bce+ranking"
                             f" or group_ranking")

        self.model_path, self.state = model_path, {}
        os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None

        if load_model and os.path.exists(model_path):
            logger.info('Load model...')
            self.load_model()

    def train_step(self, epoch: int, train_x: torch.Tensor, train_y: torch.Tensor):
        train_x = train_x.cuda()
        train_y = train_y.cuda()

        self.optimizer.zero_grad()
        self.model.train()

        if self.mixup_fn is not None and epoch >= self.mixup_warmup:
            if self.mixup_layer == 'emb':
                emb, lengths, masks = self.model(train_x, return_emb=True)
                emb, train_y = self.mixup_fn(emb, train_y)
                scores = self.model((emb, lengths, masks), pass_emb=True)
            else:
                hidden, lengths, masks = self.model(train_x, return_hidden=True)
                hidden, train_y = self.mixup_fn(hidden, train_y)
                scores = self.model((hidden, lengths, masks), pass_hidden=True)
        else:
            scores = self.model(train_x)

        loss = self.loss_fn(scores, train_y)

        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()

        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: torch.Tensor, k: int):
        self.model.eval()
        with torch.no_grad():
            scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu()

    def get_optimizer(self, **kwargs):
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer)
        # self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=100, step=100, k=5, early=50, verbose=True, swa_warmup=None,
              inv_w=None, mlb=None, criterion='ndcg5', **kwargs):
        self.nb_epoch = 100
        self.n_steps = len(train_loader)

        if inv_w is not None and mlb is not None:
            train_labels = mlb.inverse_transform(train_loader.dataset.data_y)
            valid_labels = mlb.inverse_transform(valid_loader.dataset.data_y)
            valid_mlb = MultiLabelBinarizer(sparse_output=True).fit(valid_labels)
            valid_targets = valid_mlb.transform(valid_labels)
            with contextlib.redirect_stderr(None):
                inv_w = get_inv_propensity(
                    valid_mlb.transform(train_labels),
                    inv_w.get('a', 0.55), inv_w.get('b', 1.5))
        else:
            valid_targets = None
            valid_mlb = None

        if criterion not in ['p5', 'ndcg5', 'psp5']:
            raise ValueError(f'criterion {criterion} is invalid. (p5, ndcg5, psp5)')

        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best, e = 0, 0.0, 0
        self.save_model()
        for epoch_idx in range(nb_epoch):
            if epoch_idx == swa_warmup:
                self.swa_init()
            for i, train_inputs in enumerate(train_loader, 1):
                global_step += 1
                loss = self.train_step(epoch_idx, *train_inputs)
                if global_step % step == 0:
                    self.swa_step()
                    self.swap_swa_params()
                    labels = []
                    for valid_x in valid_loader:
                        if isinstance(valid_x, (list, tuple)):
                            labels.append(self.predict_step(*valid_x, k)[1])
                        else:
                            labels.append(self.predict_step(valid_x, k)[1])
                    labels = np.concatenate(labels)
                    targets = valid_loader.dataset.data_y
                    p5, n5 = get_p_5(labels, targets), get_n_5(labels, targets)
                    if inv_w is not None and mlb is not None:
                        with contextlib.redirect_stderr(None):
                            psp5 = get_psp_5(mlb.classes_[labels], valid_targets, inv_w,
                                            valid_mlb)
                    else:
                        psp5 = None

                    if criterion == 'p5':
                        cur = p5
                    elif criterion == 'ndcg5':
                        cur = n5
                    else:
                        cur = psp5

                    if cur > best:
                        self.save_model()
                        best, e = cur, 0
                    else:
                        e += 1
                        if early is not None and e > early:
                            return
                    self.swap_swa_params()
                    if verbose:
                        logger.info(F'{epoch_idx} {i * train_loader.batch_size} train loss: {round(loss, 5)} '
                                    F'P@5: {round(p5, 5)} nDCG@5: {round(n5, 5)} PSP@5: {round(psp5, 5)} '
                                    F'early stop: {e}')

    def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
        self.load_model()
        scores_list = []
        labels_list = []
        for data_x in tqdm(data_loader, desc=desc, leave=False):
            if isinstance(data_x, (list, tuple)):
                outputs = self.predict_step(*data_x, k)
            else:
                outputs = self.predict_step(data_x, k)
            scores_list.append(outputs[0])
            labels_list.append(outputs[1])

        return np.concatenate(scores_list), np.concatenate(labels_list)

    def save_model(self):
        torch.save(self.model.module.state_dict(), self.model_path)

    def load_model(self):
        self.model.module.load_state_dict(torch.load(self.model_path))

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warning(F'Clipping gradients with total norm {round(total_norm.item(), 5)} '
                               F'and max norm {round(max_norm.item(), 5)}')

    def swa_init(self):
        if 'swa' not in self.state:
            logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.clone().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n], p.data

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']


class XMLModel(Model):
    """

    """
    def __init__(self, labels_num, hidden_size, device_ids=None, attn_device_ids=None,
                 most_labels_parallel_attn=80000, **kwargs):
        parallel_attn = labels_num <= most_labels_parallel_attn
        load_model = kwargs.pop('load_model', False)
        super(XMLModel, self).__init__(hidden_size=hidden_size, device_ids=device_ids, labels_num=labels_num,
                                       parallel_attn=parallel_attn, load_model=False, **kwargs)
        self.network, self.attn_weights = self.model, nn.Sequential()
        if not parallel_attn:
            self.attn_weights = AttentionWeights(labels_num, hidden_size*2, attn_device_ids)
        self.model = nn.ModuleDict({'Network': self.network.module, 'AttentionWeights': self.attn_weights})
        self.state['best'] = {}

        if load_model:
            self.load_model()

    def train_step(self, epoch: int, train_x: Tuple[torch.Tensor, torch.Tensor],
                   train_y: torch.Tensor):
        self.optimizer.zero_grad()
        train_x, candidates = train_x
        self.model.train()
        scores = self.network(train_x, candidates=candidates, attn_weights=self.attn_weights)
        loss = self.loss_fn(scores, train_y)
        loss.backward()
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], k):
        data_x, candidates, group_scores = data_x
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.network(data_x, candidates=candidates, attn_weights=self.attn_weights))
            scores, labels = torch.topk(scores * group_scores.cuda(), k)
            return scores.cpu(), candidates[np.arange(len(data_x)).reshape(-1, 1), labels.cpu()]

    def train(self, *args, **kwargs):
        super(XMLModel, self).train(*args, **kwargs)
        self.save_model_to_disk()

    def save_model(self):
        model_dict = self.model.state_dict()
        for key in model_dict:
            self.state['best'][key] = model_dict[key].cpu().detach()

    def save_model_to_disk(self):
        model_dict = self.model.state_dict()
        for key in model_dict:
            model_dict[key][:] = self.state['best'][key]
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location='cuda:0'))


class TransformerXML(Model):
    def get_optimizer(self, **kwargs):
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.9,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        scheduler_opt = kwargs.pop('scheduler', None)
        self.optimizer = AdamW(param_groups, **kwargs)
        self.scheduler = None

        if scheduler_opt is not None:
            num_warmup_steps = scheduler_opt.get('warmup', 5) * self.n_steps
            num_training_steps = self.nb_epoch * self.n_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps,
                                                             num_training_steps)
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer)
        # self.model = nn.DataParallel(self.model, device_ids=self.device_ids)


    def train_step(self, epoch: int, train_x: torch.Tensor,
                   attention_mask: torch.Tensor, train_y: torch.Tensor):
        train_x = train_x.cuda()
        attention_mask = attention_mask.cuda()
        train_y = train_y.cuda()

        self.optimizer.zero_grad()
        self.model.train()

        if self.mixup_fn is not None and epoch >= self.mixup_warmup:
            outputs = self.model(train_x, attention_mask, return_hidden=True)
            hidden, train_y = self.mixup_fn(outputs[0], train_y)
            logits = self.model(pass_hidden=True, outputs=(hidden, *outputs[1:]))[0]
        else:
            logits = self.model(train_x, attention_mask)[0]
        loss = self.loss_fn(logits, train_y)

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def predict_step(self, data_x: torch.Tensor, attention_mask: torch.Tensor, k: int):
        data_x = data_x.cuda()
        attention_mask = attention_mask.cuda()

        self.model.eval()
        with torch.no_grad():
            scores, labels = torch.topk(self.model(data_x, attention_mask)[0], k)
            return torch.sigmoid(scores).cpu(), labels.cpu()
