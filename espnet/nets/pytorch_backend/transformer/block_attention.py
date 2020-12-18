#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn
import logging

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    
class RelBlockMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate, pos_emb, block_len=None):
        super(RelBlockMultiHeadedAttention, self).__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        if block_len is not None:
            self.block_len = block_len
        else:
            self.block_len = 0

        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_r = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

        self.bu = nn.Parameter(torch.Tensor(self.h, self.d_k).uniform_(0, 1))
        self.bv = nn.Parameter(torch.Tensor(self.h, self.d_k).uniform_(0, 1))

        self.pos_emb = pos_emb
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask, key_memory=None, key_memory_mask=None, block_len=None, kmem_after=False):

        qlen = query.size(1)
        n_batch, klen, dim = key.size()
        if block_len is not None:
            self.block_len = block_len
        if self.block_len > 0:
            if query.size(1) != key.size(1):
                print("lenght of query and key must be same when block processing is enabled")
                exit(-1)
            blen = self.block_len
            if klen % blen > 0:
                plen = blen - klen % blen
                klen = klen + plen
            else:
                plen = 0

            q = self.linear_q(query)
            q = torch.nn.functional.pad(q, (0, 0, 0, plen))
            q = q.view(-1, blen, self.h, self.d_k)
            
            k = self.linear_k(key)
            k = torch.nn.functional.pad(k, (0, 0, blen, plen))
            k = k.as_strided((n_batch, int(klen / blen), blen * 2, dim),
                             ((blen + klen) * dim, dim * blen, dim, 1))
            kmem_len = 0
            if key_memory is not None:
                #key_memory = key_memory.detach()
                key_memory = self.linear_k(key_memory)
                kmem_len = key_memory.size(1)
                kmem_ext = torch.zeros(n_batch, int(klen / blen), kmem_len, self.h*self.d_k, device=k.device)
                kmem_ext = kmem_ext.copy_(key_memory.unsqueeze(1))
                #k = torch.cat([k, kmem_ext], dim=-2)
                if kmem_after:
                    k = torch.cat([k, kmem_ext], dim=-2)
                else:
                    k = torch.cat([kmem_ext, k], dim=-2)
            k = k.contiguous().view(-1, blen * 2 + kmem_len , self.h, self.d_k)
            #k[:, (kmem_len + blen):, :, :] = k[:, (kmem_len + blen):, :, :].detach()#.clone()

            v = self.linear_v(value)
            v = torch.nn.functional.pad(v, (0, 0, blen, plen))
            v = v.as_strided((n_batch, int(klen / blen), blen * 2, dim),
                             ((blen + klen) * dim, dim * blen,  dim, 1))            
            if key_memory is not None: # assuming key_memory is identical to value. should be separated?
                kmem_len = key_memory.size(1)
                kmem_ext = torch.zeros(n_batch, int(klen / blen), kmem_len, self.h*self.d_k, device=k.device)
                kmem_ext = kmem_ext.copy_(key_memory.unsqueeze(1))
                #v = torch.cat([v, kmem_ext], dim=-2)
                if kmem_after:
                    v = torch.cat([v, kmem_ext], dim=-2)
                else:
                    v = torch.cat([kmem_ext, v], dim=-2)
            v = v.contiguous().view(-1, blen * 2 + kmem_len, self.h, self.d_k)
            #v[:, (kmem_len + blen):, :, :] = v[:, (kmem_len + blen):, :, :].detach()#.clone()

            # handling of masks might be insufficient
            if mask is not None:
                mask = torch.nn.functional.pad(mask.squeeze(1), (blen, plen, 0, 0)) # B x t1(q) x t2(k)
                mask = mask.as_strided((n_batch, int(klen / blen), blen * 2),
                                       ((blen + klen), blen, 1))
                if kmem_len > 0:
                    if key_memory_mask is not None:
                        kmask_ext = torch.zeros(n_batch, int(klen / blen), kmem_len, dtype=key_memory_mask.dtype, device=key_memory_mask.device)
                        kmask_ext = kmask_ext.copy_(key_memory_mask)
                        if  kmem_after:
                            mask = torch.cat([mask, kmask_ext], dim=-1)
                        else:
                            mask = torch.cat([kmask_ext, mask], dim=-1)
                    else:
                        print("key_memory_mask must be given")
                        exit(-1)
                mask = mask.contiguous().view(-1, blen * 2 + kmem_len)
                mask = mask.unsqueeze(1)

        else:
            if key_memory is not None:
                #key_memory = key_memory.detach()
                if kmem_after:
                    key = torch.cat([key, key_memory], dim=-2)
                    value = torch.cat([value, key_memory], dim=-2)                
                else:
                    key = torch.cat([key_memory, key], dim=-2)
                    value = torch.cat([key_memory, value], dim=-2)                
            q = self.linear_q(query)
            k = self.linear_k(key)
            v = self.linear_v(value)

            if key_memory_mask is not None:
                if kmem_after:
                    mask = torch.cat([mask, key_memory_mask], dim=-1)
                else:
                    mask = torch.cat([key_memory_mask, mask], dim=-1)

            q = q.view(n_batch, -1, self.h, self.d_k)
            k = k.view(n_batch, -1, self.h, self.d_k)
            v = v.view(n_batch, -1, self.h, self.d_k)

        klen = k.size()[1]
        r = torch.zeros(self.d_k * self.h * klen, device=q.device)
        r = r.view(klen, self.d_k * self.h).unsqueeze(0)
        r = self.linear_r(self.pos_emb(r)).view(klen, self.h, self.d_k)
        
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        r = r.unsqueeze(0).transpose(1, 2) # (batch, head, time2, d_k)

        bu = self.bu.unsqueeze(0).unsqueeze(2)
        t_ac = torch.matmul((q + bu), k.transpose(-2, -1))
        bv = self.bv.unsqueeze(0).unsqueeze(2)
        t_bd = torch.matmul((q + bv), r.transpose(-2, -1))

        ## rel shift of mine
        f_t_bd = torch.flip(t_bd, dims=[-1])[:,:,:,1:]
        B, H, T1, T2 = t_bd.size()
        if kmem_after:
            ofset = T2 - 1
        else:
            ofset = T1 - 1
        t_bd = torch.cat([t_bd, f_t_bd], dim=-1).as_strided((B,H,T1,T2), (H*T1*(T2*2-1),T1*(T2*2-1),T2*2-2,1), storage_offset=ofset)
        ## rel shift of ESPnet
        # zero_pad = torch.zeros((t_bd.size(0), 1, *t_bd.size()[2:]), device=t_bd.device, dtype=t_bd.dtype)
        # t_bd_padded = torch.cat([zero_pad, t_bd], dim=1)
        # t_bd_padded = t_bd_padded.view(t_bd.size(1) + 1, t_bd.size(0), *t_bd.size()[2:])
        # t_bd = t_bd_padded[1:].view_as(t_bd)
        
        scores = (t_ac + t_bd) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)[:, :qlen, :]  # (batch, time1, d_model)
