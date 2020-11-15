# -*- coding: utf-8
"""
Created on 2020/11/11
@author uoo723

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FocalLoss', "RankingLoss", "CombinedLoss", "GroupRankingLoss"]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be `mean` or `sum` or `none`")

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  reduction='none')
        pt = torch.exp(-loss)
        f_loss = (1 - pt) ** self.gamma * loss

        if self.weight is not None:
            f_loss = self.weight.unsqueeze(1) * f_loss

        if self.pos_weight is not None:
            f_loss = self.pos_weight * f_loss

        if self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:
            return f_loss


# https://openreview.net/pdf?id=hUAmiQCeUGm
# RankNet
class RankingLoss(nn.Module):
    def __init__(self, freq, gamma=0.1, c=0.0):
        super(RankingLoss, self).__init__()
        self.freq = freq
        self.gamma = gamma
        self.c = c

    def forward(self, input, target):
        input = torch.sigmoid(input)
        if not torch.is_tensor(self.freq):
            self.freq = torch.tensor(self.freq, device=input.device, dtype=torch.float32)

        loss = 0

        for x, y in zip(input, target):
            x_pos = x.masked_select(y.byte())
            x_neg = x.masked_select(~y.byte())

            x_rolled = torch.stack([x_pos.roll(shifts=i) for i in range(x_pos.shape[0])])
            cnt = self.freq[y.nonzero().view(-1)]
            cnt2 = torch.stack([cnt.roll(shifts=i) for i in range(cnt.shape[0])])
            n_matrix = (cnt - cnt2) / (cnt + cnt2)

            loss1 = torch.sum(torch.clamp(n_matrix * (x_pos - x_rolled), min=0))
            loss2 = self.gamma * torch.sum(torch.clamp(x_neg - x_pos.unsqueeze(dim=1) + self.c, min=0))

            loss = loss + loss1 + loss2

        loss = loss / input.shape[0]

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, *loss_fn):
        super(CombinedLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target):
        return torch.stack([fn(input, target) for fn in self.loss_fn]).sum()


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970478
# Group wise ranking loss
class GroupRankingLoss(nn.Module):
    def forward(self, input, target):
        input = torch.sigmoid(input)
        sorted_input, _ = input.sort(descending=True)
        counts = target.sum(dim=-1)
        i = torch.stack([torch.arange(counts.size(0), dtype=torch.long,
                                      device=target.device), counts.long() - 1])
        lamda = sorted_input[i[0, :], i[1, :]]

        a = lamda * counts
        b = torch.clamp(input - lamda.unsqueeze(dim=-1), min=0).sum(dim=-1)
        c = (input * target).sum(dim=-1)

        loss = (a + b - c) / counts
        return loss.mean()
