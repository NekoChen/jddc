'''Attention'''
#Attention函数可以将Query和一组Key-Value对映射到输出,
# 其中Query、Key、Value和输出都是向量。
# 输出是值的加权和，其中分配给每个Value的权重由Query与相应Key的兼容函数计算
# 输入包含维度dk的Query和Key，以及维度dv的Value
# 首先计算Query和各个Key的点积，然后除以根号dk，用Softmax获得Key的权重
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value,mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        attn = torch.matmul(query,key.transpose(-2,-1)) \
               / math.sqrt(d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(attn, dim = -1)
        if self.attn_dropout is not None:
            p_attn = self.attn_dropout(p_attn)
        output = torch.matmul(p_attn, value)
        return output, p_attn
