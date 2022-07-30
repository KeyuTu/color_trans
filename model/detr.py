import copy
import random
from typing import Optional, List

import sys
sys.path.insert(0, '/ghome/zhuangjf/.local/lib/python3.7/site-packages')

import torch
import torch.nn.functional as F
from torch import nn, Tensor, einsum


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, dropout=0.):
        super().__init__()

        # self.scale = dim_head**-0.5
        self.scale = 7
        #print(self.scale)
        #exit()

        if True:
            self.to_q = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim_head, bias=False),
                                  nn.Linear(dim_head, dim_head, bias=False))
            self.to_k = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim_head, bias=False),
                                  nn.Linear(dim_head, dim_head, bias=False))
            self.to_v = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim_head, bias=False),
                                  nn.Linear(dim_head, dim_head, bias=False))
        else:
            self.to_q = nn.Linear(dim, dim_head, bias=False)
            self.to_k = nn.Linear(dim, dim_head, bias=False)
            self.to_v = nn.Linear(dim, dim_head, bias=False) 

        self.to_out = nn.Sequential(nn.Linear(dim_head, dim), nn.Dropout(dropout))
        #print('ppppppp')
        #exit()
    def forward(self, query, key, value,target=None):
        b, n_q, c = query.shape
        _, n_k, _ = key.shape
        #dots = einsum('b i d, b j d -> b i j', query, key) * self.scale
        #print(dots[0,0,:])
        #exit()
        #attn = dots.softmax(dim=-1)
        #print(attn[0,0])
        #index=attn.data.max(2)[1]
        #print(index)
        #print(target)
        #print(attn.max(2)[0])   
        #exit()
        #print(query.shape,key.shape,value.shape)
        query = self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value)
        #print(query.shape,key.shape,value.shape)
        #exit()
        dots = einsum('b i d, b j d -> b i j', query, key) * self.scale
        attn = dots.softmax(dim=-1)
        #print(attn[0,0])
        #exit()
        out = einsum('b i j, b j d -> b i d', attn, value)
        out = self.to_out(out)

        return out


class TransformerHELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim=d_model, dim_head=dim_feedforward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, query, key, value,target=None):
        query_ori=query
        #print(key.shape,value.shape)
        #exit()
        query2 = self.attn(query=query, key=key, value=value,target=target)
        query = query + self.dropout(query2)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout1(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query =self.norm2(query)
        #query=F.normalize(query)
        return query


if __name__ == '__main__':
    dec_layer = TransformerHELayer(d_model=128, dim_feedforward=128)
    dec_layer.cuda()

    query = torch.rand([1, 1, 128]).cuda()
    key = torch.rand([1, 190, 128]).cuda()
    value = torch.rand([1, 190, 128]).cuda()
    out = dec_layer(query=query, key=key, value=value)
    print(out.shape)