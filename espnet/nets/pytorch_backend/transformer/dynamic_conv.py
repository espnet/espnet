import math
import logging

import numpy
import torch
import torch.nn.functional as F
from torch import nn


MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class DynamicConvolution(nn.Module):
    """Dynamic Convolution layer
    This implementation is based on https://github.com/pytorch/fairseq/tree/master/fairseq

    :param int whare: the number of kernel of convolution
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, wshare, n_feat, dropout_rate, kernel_size_str, lnum, use_kernel_mask=False,use_bias=False):
        super(DynamicConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = int(kernel_size_str.split("-")[lnum])
        self.padding_size = int(self.kernel_size/2)
        self.attn = None

        # linear -> GLU -- -> lightconv -> linear 
        #               \        /
        #                 Lienar
        self.linear1 = nn.Linear(n_feat,n_feat*2)
        self.linear2 = nn.Linear(n_feat,n_feat)
        self.linear_weight = nn.Linear(n_feat,self.wshare*1*self.kernel_size)
        nn.init.xavier_uniform(self.linear_weight.weight)
        self.act = nn.GLU()

        #dynamic conv related
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat)) 

    def forward(self, query, key, value, mask):
        """Compute 'Dynamic Convolution'
        This function takes query, key and value but uses only value. This is just for compatibility with self-attention layer (attention.py)
        :param torch.Tensor mask: (batch, time1)
        :return torch.Tensor: convoluved `value` (batch, time1, d_model)
        """

        # linear -> GLU -- -> lightconv -> linear 
        #               \        /
        #                 Lienar
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size
        
        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)
    
        # get kernel of convolution
        weight = self.linear_weight(x) # B x T x kH
        weight = F.dropout(weight,self.dropout_rate,training=self.training)
        weight = weight.view(B,T,H,k).transpose(1,2).contiguous() # B x H x T x k
        weight_new = torch.zeros(B*H*T*(T+k-1)).view(B,H,T,T+k-1).fill_(float('-inf')) # B x H x T x T+k-1
        weight_new = weight_new.to(x.device)
        weight_new.as_strided((B,H,T,k),((T+k-1)*T*H, (T+k-1)*T, T+k, 1)).copy_(weight)
        weight_new = weight_new.narrow(-1,int((k-1)/2),T) # B x H x T x T(k)
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask==0.0, float('-inf'))
        weight_new = F.softmax(weight_new,dim=-1)
        self.attn = weight_new
        weight_new = weight_new.view(B*H,T,T)
        
        # convolution
        x = x.transpose(1,2).contiguous() # B x C x T
        x = x.view(B*H,int(C/H),T).transpose(1,2)
        x = torch.bmm(weight_new,x) # BH x T x C/H
        x = x.transpose(1,2).contiguous().view(B, C, T)

        if self.use_bias:
            x = x + self.bias.view(1,-1,1)
        x = x.transpose(1,2) # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1,-2)
            x = x.masked_fill(mask == 0, 0.0)

        # second linear layer
        x = self.linear2(x)
        return x  
