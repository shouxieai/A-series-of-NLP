import torch.nn.functional as F
import math
import torch
import os
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np

class PositionEncoder(nn.Module):
    def __init__(self, embedding_num, max_len, device, dropout=0.3):
        super().__init__()

        self.embedding_num = embedding_num
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.embedding_num, device=device)

        position = torch.arange(1, self.max_len+1, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.embedding_num, 2).float() * (-math.log(10000.0) / self.embedding_num))
        div_term = torch.rand(size=(50,)).sort(descending=True)[0]
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        # self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.shape[1]]
        x = self.dropout(x)

        return x


def get_attn_pad_mask(len_list, device):
    max_len = max(len_list)
    batch_size = len(len_list)

    pad_attn_mask = torch.ones(size=(batch_size, max_len, max_len), dtype=torch.int8, device=device)
    for index, i in enumerate(len_list):
        pad_attn_mask[index][:i, :i] = 0
    return pad_attn_mask


class EncoderBlock(nn.Module):
    def __init__(self, embedding_num, n_heads, ff_num):
        super().__init__()
        self.embedding_num = embedding_num
        self.n_heads = n_heads

        # --------------------------  MultiHeadAttention -------------------
        self.W_Q = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.W_K = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.W_V = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.fc = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.att_ln = nn.LayerNorm(self.embedding_num)

        # --------------------------- FeedForward  --------------------------
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_num, ff_num, bias=False),
            nn.ReLU(),
            nn.Linear(ff_num, self.embedding_num, bias=False)
        )
        self.feed_ln = nn.LayerNorm(self.embedding_num)

    def forward(self, x, attn_mask):
        # ---------------------- MultiHeadAttention forward ------------------
        Q = self.W_Q(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
        K = self.W_K(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
        V = self.W_K(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
        attn_mask_new = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.embedding_num / self.n_heads)
        # scores.masked_fill_(attn_mask_new.type(torch.bool), -1e9)

        attn = F.softmax(scores, dim=-1)
        context = (attn @ V).transpose(1, 2).reshape(*x.shape)

        att_result = self.fc(context)
        att_result = self.att_ln(x + att_result)

        #  ----------------------- FeedForward forward --------------------
        feed_result = self.feed_forward(att_result)
        feed_result = self.feed_ln(att_result + feed_result)

        return feed_result, attn


class TransformerEncoder(nn.Module):
    def __init__(self,device="cpu",block_nums=2, embedding_num=200, max_len=1000, n_heads=2, ff_num=128):
        super().__init__()
        self.device = device
        # ----------------------- position encoder --------------------------
        self.position_encoder = PositionEncoder(embedding_num, max_len, device)

        # ---------------------------- blocks -------------------------
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embedding_num, n_heads, ff_num) for _ in range(block_nums)])

        # ------------------------ classifiy and Loss------------------------
        # self.classifier = nn.Linear(embedding_num, params["class_num"])
        # self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, batch_x, datas_len):
        # ------------------- position embedding -----------------
        encoder_output = self.position_encoder.forward(batch_x)

        # -------------------- blocks forward --------------------
        enc_self_attn_mask = get_attn_pad_mask(datas_len, self.device)
        for block in self.encoder_blocks:
            encoder_output, _ = block.forward(encoder_output, enc_self_attn_mask)

        return encoder_output

        # # ----------------- output and predict ----------------
        # encoder_output = torch.max(encoder_output,dim=1)
        # pre = self.classifier(encoder_output)
        # if batch_label is not None:
        #     return self.cross_loss(pre,batch_label)
        # else:
        #     return torch.argmax(pre,dim=-1)