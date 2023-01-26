# -*- coding: utf-8 -*-
"""
@Time ： 2023/1/23 18:37
@Auth ： Murphy
@File ：my_loss.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
class DomainDisentangleLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(DomainDisentangleLoss, self).__init__()
        self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()

        self.alpha1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) if weight is None else weight
        self.alpha2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # weights are hyperparameters
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1


    def forward(self, fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y, yd): # 这个input本身就是model处理的结果所以不用将model 传入loss类了

        L_class = self.nll_loss(torch.log(Cfcs),y) + self.alpha1*self.entropy_loss(DCfcs)
        L_domain = self.cross_entropy_loss(DCfds,yd) + self.alpha2*self.entropy_loss(Cfds)
        # print(self.entropy_loss(DCfcs))
        tot_loss = self.w1 * L_class + self.w2 * L_domain  + self.w3 * self.rec_loss(fG,fG_hat)
        return tot_loss

    def entropy_loss(self, f):  # 应该返回一个标量 最后是求和的
        #  ?Freeze?
        # print(f)
        f = f+0.0001 # 防止出现log(0)
        logf = torch.log(f)
        mlogf = logf.mean(dim=0)
        # l = mlogf.sum()
        return -mlogf.sum()

