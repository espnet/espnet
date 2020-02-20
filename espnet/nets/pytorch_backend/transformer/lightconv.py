import math

import numpy
import torch
import torch.nn.functional as F
from torch import nn


MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class LightweightConvolution(nn.Module):
    """Lightweight Convolution layer
    This implementation is based on https://github.com/pytorch/fairseq/tree/master/fairseq

    :param int whare: the number of kernel of convolution
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, wshare, n_feat, dropout_rate, kernel_size_str, lnum, use_kernel_mask=False,use_bias=False):
        super(LightweightConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = int(kernel_size_str.split("-")[lnum])
        self.padding_size = int(self.kernel_size/2)

        # linear -> GLU -> lightconv -> linear
        self.linear1 = nn.Linear(n_feat,n_feat*2)
        self.linear2 = nn.Linear(n_feat,n_feat)
        self.act = nn.GLU()

        #lightconv related
        self.weight = nn.Parameter(torch.Tensor(self.wshare, 1, self.kernel_size).uniform_(0,1))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat)) 

        # mask of kernel
        kernel_mask0 = torch.zeros(self.wshare, int(self.kernel_size/2))
        kernel_mask1 = torch.ones(self.wshare, int(self.kernel_size/2+1))
        self.kernel_mask = torch.cat((kernel_mask1, kernel_mask0), dim=-1).unsqueeze(1)

    def forward(self, query, key, value, mask):
        """Compute 'Lightweight Convolution'
        This function takes query, key and value but uses only value. This is just for compatibility with self-attention layer (attention.py)
        :param torch.Tensor mask: (batch, time1)
        :return torch.Tensor: convoluved `value` (batch, time1, d_model)
        """

        # linear -> GLU -> lightconv -> linear
        x = query
        B, T, C = x.size()
        H = self.wshare
        
        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # lightconv
        x = x.transpose(1,2).contiguous().view(-1, H, T) # B x C x T
        weight = F.dropout(self.weight,self.dropout_rate,training=self.training)
        if self.use_kernel_mask:
            self.kernel_mask = self.kernel_mask.to(x.device)
            weight = weight.masked_fill(self.kernel_mask==0.0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        x = F.conv1d(x, weight, padding=self.padding_size, groups=self.wshare).view(B, C, T)
        if self.use_bias:
            x = x + self.bias.view(1,-1,1)
        x = x.transpose(1,2) # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1,-2)
            x = x.masked_fill(mask == 0, 0.0)

        # second linear layer
        x = self.linear2(x)
        return x  
