#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""
import torch.nn as nn
import torch.nn.functional as F

from deepxml.modules import *

__all__ = ['AttentionRNN', 'FastAttentionRNN']


class Network(nn.Module):
    """

    """
    def __init__(self, emb_size, vocab_size=None, emb_init=None, emb_trainable=True,
                 padding_idx=0, emb_dropout=0.2, **kwargs):
        super(Network, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable,
                             padding_idx, emb_dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num,
                 linear_size, dropout, **kwargs):
        super(AttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = MLAttention(labels_num, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(self, inputs, return_emb=False, pass_emb=False,
                return_hidden=False, pass_hidden=False, **kwargs):
        if return_emb and pass_emb:
            raise ValueError("`return_emb` and `pass_emb` both cannot be True")

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_emb and return_hidden:
            raise ValueError("`return_emb` and `return_hidden` both cannot be True")

        if not pass_emb and not pass_hidden:
            emb_out, lengths, masks = self.emb(inputs, **kwargs)
        elif not pass_hidden:
            emb_out, lengths, masks = inputs
        else:
            emb_out, lengths, masks = None, None, None

        if return_emb:
            return emb_out, lengths, masks

        if emb_out is not None:
            emb_out, masks = emb_out[:, :lengths.max()], masks[:, :lengths.max()]

        if not pass_hidden:
            rnn_out = self.lstm(emb_out, lengths)       # N, L, hidden_size * 2
        else:
            rnn_out = inputs

        if return_hidden:
            return rnn_out

        attn_out = self.attention(rnn_out, masks)   # N, labels_num, hidden_size * 2

        return self.linear(attn_out)


class FastAttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, parallel_attn, **kwargs):
        super(FastAttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = FastMLAttention(labels_num, hidden_size * 2, parallel_attn)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(self, inputs, candidates, attn_weights: nn.Module, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks, candidates, attn_weights)     # N, sampled_size, hidden_size * 2
        return self.linear(attn_out)


# https://github.com/tkipf/pygcn/blob/master/pygcn/models.py
class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
