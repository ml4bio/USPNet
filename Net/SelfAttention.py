import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter, init

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    # query ,key, value shape: (batch_size, h, seq_length, d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # scores shape: (batch_size, h, seq_length, seq_length)
    # p_attn shape: (batch_size, h, seq_length, seq_length)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        d_model = config['d_model']
        h = config['h']
        dropout = config['dropout']
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        # print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


if __name__ == '__main__':
    h = 8
    d_model = 512
    batch_size = 1
    seq_length = 10
    model = MultiHeadAttention(h, d_model)

    query = torch.randn([batch_size, seq_length, d_model])
    key = query
    value = query

    print('Input size: ' + str(query.size()))

    m = model(query, key, value)

    print('Output size: ' + str(m.size()))