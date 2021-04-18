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
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # 里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
        # Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        # 也就是让某个神经元的激活值以一定的概率p，让其停止工作，
        # 这次训练过程中不更新权值，也不参加神经网络的计算

    def forward(self, q, k, v, mask=None):
        # 为什么k.transpose(2, 3)待定
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn