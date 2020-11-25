#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import (RobertaClassificationHead,
                                           RobertaModel,
                                           RobertaPreTrainedModel)

from deepxml.modules import *

__all__ = ['AttentionRNN', 'FastAttentionRNN', 'RobertaForSeqClassification']


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
            rnn_out, lengths, masks = inputs

        if return_hidden:
            return rnn_out, lengths, masks

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


# https://huggingface.co/transformers/_modules/transformers/modeling_roberta.html#RobertaForSequenceClassification
# Reimplementation for mix-up
class RobertaForSeqClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_hidden=False,
        pass_hidden=False,
        outputs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` cannot be both true.")

        if not pass_hidden:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
        else:
            sequence_output = outputs[0]

        if return_hidden:
            return outputs

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
