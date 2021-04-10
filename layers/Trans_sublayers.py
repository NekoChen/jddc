''' Define the sublayers in encoder/decoder layer '''
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .Trans_modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    # “多头”机制能让模型考虑到不同位置的Attention
    # 另外“多头”Attention可以在不同的子空间表示不一样的关联关系
    # MultiHead(Q, K, V) = Concat(head1, ... , headh)Wo
    # headi = attn(QWq, KWk, VWv)
    def __init__(self,n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    # 全连接前馈网络
    # 包括两个线性变换+ReLu作为激活
    def __init__(self, d_model, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_model) # position-wise
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x



# class MultiHeadedAttention(nn.Module):
#     # “多头”机制能让模型考虑到不同位置的Attention
#     # 另外“多头”Attention可以在不同的子空间表示不一样的关联关系
#     # MultiHead(Q, K, V) = Concat(head1, ... , headh)Wo
#     # headi = attn(QWq, KWk, VWv)
#
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = self.clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def clones(module, N):
#         "Produce N identical layers."
#         return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = ScaledDotProductAttention(query, key, value, mask=mask,
#                                  dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)