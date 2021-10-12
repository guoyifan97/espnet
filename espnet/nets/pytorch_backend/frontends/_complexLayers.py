#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
import numpy
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv1d, Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from espnet.nets.pytorch_backend.frontends._complexFunctions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_avg_pool2d
from espnet.nets.pytorch_backend.frontends._complexFunctions import complex_dropout, complex_dropout2d
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List


class ComplexMultiSequential(Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """Repeat module N times.

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(N)])


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class ComplexPositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position

    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False, eps=0.01):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        if d_model & 2:
            self.xscale = math.sqrt(self.d_model-1)
        else:
            self.xscale = math.sqrt(self.d_model)
        self.dropout = ComplexDropout(p=dropout_rate)
        self.pe_r, self.pe_i = None, None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len), torch.tensor(0.0).expand(1, max_len), eps=eps)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x_r, x_i, eps=0.01):
        """Reset the positional encodings."""
        
        if self.pe_r is not None:
            assert self.pe_r.shape == self.pe_i.shape and x_r.shape == x_i.shape
            assert self.pe_r.dtype == self.pe_i.dtype and x_r.dtype == x_i.dtype
            assert self.pe_r.device == self.pe_i.device and x_r.device == x_i.device
            if self.pe_r.size(1) >= x_r.size(1):
                if self.pe_r.dtype != x_r.dtype or self.pe_r.device != x_r.device:
                    self.pe_r = self.pe_r.to(dtype=x_r.dtype, device=x_r.device)
                    self.pe_i = self.pe_i.to(dtype=x_i.dtype, device=x_i.device)
                return
        
        pe_r, pe_i = torch.zeros(x_r.size(1), self.d_model), torch.zeros(x_i.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x_r.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x_r.size(1), dtype=torch.float32).unsqueeze(1)

        # div_term (0, 2, ..., 255), d_model is odd
        if self.d_model % 2:
            div_term = torch.exp(
                torch.arange(0, self.d_model-1, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        else:
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )

        # 126
        pe_cos = torch.cos(position * div_term + eps)
        pe_sin = torch.sin(position * div_term + eps)

        # 126
        if self.d_model % 2:
            pe_r[:, 0:-1:2], pe_i[:, 0:-1:2] = pe_cos, pe_sin
        else:
            pe_r[:, 0::2], pe_i[:, 0::2] = pe_cos, pe_sin
        # 126
        pe_r[:, 1::2], pe_i[:, 1::2] = pe_cos, -pe_sin
        pe_r, pe_i = pe_r.unsqueeze(0), pe_i.unsqueeze(0)
        self.pe_r, self.pe_i = pe_r.to(device=x_r.device, dtype=x_r.dtype), pe_i.to(device=x_i.device, dtype=x_i.dtype)

    def forward(self, x_r, x_i):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, C, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, C, ...)

        """
        self.extend_pe(x_r, x_i)
        x_r, x_i = x_r * self.xscale + self.pe_r[:, : x_r.size(1)], x_i * self.xscale + self.pe_i[:, : x_i.size(1)]
        return self.dropout(x_r, x_i)


class ComplexPositionalEncoding2D(torch.nn.Module):
    """Positional encoding for Universal Beamformer for BCTF.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_time_len: maximum input time length
    :param reverse: whether to reverse the input position

    """

    def __init__(self, d_model, dropout_rate, max_channel_num=5, max_time_len=1000, reverse=False, omit_channel=False, eps=0.01):
        """Construct an PositionalEncoding object."""

        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        if d_model % 2:
            self.xscale = math.sqrt(self.d_model-1)
        else:
            self.xscale = math.sqrt(self.d_model)
        self.dropout = ComplexDropout(p=dropout_rate)
        self.pe_r, self.pe_i = None, None
        self.omit_channel = omit_channel
        self.extend_pe(torch.tensor(0.0).expand(1, max_channel_num, max_time_len), torch.tensor(0.0).expand(1, max_channel_num, max_time_len), eps=eps)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x_r, x_i, eps=0.01):
        """Reset the positional encodings.
            x_r, x_i torch.Tensor (B, C, T, F)
        """
        
        if self.pe_r is not None:
            assert self.pe_r.shape == self.pe_i.shape and x_r.shape == x_i.shape
            assert self.pe_r.dtype == self.pe_i.dtype and x_r.dtype == x_i.dtype
            assert self.pe_r.device == self.pe_i.device and x_r.device == x_i.device
            if self.pe_r.size(1) >= x_r.size(1) and self.pe_r.size(2) >= x_r.size(2):
                if self.pe_r.dtype != x_r.dtype or self.pe_r.device != x_r.device:
                    self.pe_r = self.pe_r.to(dtype=x_r.dtype, device=x_r.device)
                    self.pe_i = self.pe_i.to(dtype=x_i.dtype, device=x_i.device)
                return
        
        pe_r, pe_i = torch.zeros(x_r.size(1), x_r.size(2), self.d_model), torch.zeros(x_i.size(1), x_r.size(2), self.d_model)
        if self.reverse:
            raise
            position = torch.arange(
                x_r.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            # CT1
            if self.omit_channel:
                # position = torch.ger(torch.ones(x_r.size(1)), torch.arange(1, x_r.size(2)+1, dtype=torch.float32)).unsqueeze(-1)
                position = torch.arange(x_r.size(2), dtype=torch.float32).unsqueeze(0).expand(x_r.size(1), x_r.size(2)).unsqueeze(-1)
            else:
                position = (10*torch.arange(x_r.size(1), dtype=torch.float32).unsqueeze(1) + torch.arange(x_r.size(2), dtype=torch.float32)).unsqueeze(-1)

        # div_term (0, 2, ..., 255), d_model is odd
        if self.d_model % 2:
            div_term = torch.exp(
                torch.arange(0, self.d_model-1, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        else:
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )

        # CTF
        pe_cos = torch.cos(position * div_term + eps)
        pe_sin = torch.sin(position * div_term + eps)
        # 126
        if self.d_model % 2:
            pe_r[:, :, 0:-1:2], pe_i[:, :, 0:-1:2] = pe_cos, pe_sin
        else:
            pe_r[:, :, 0::2], pe_i[:, :, 0::2] = pe_cos, pe_sin
        # 126
        pe_r[:, :, 1::2], pe_i[:, :, 1::2] = pe_cos, -pe_sin
        pe_r, pe_i = pe_r.unsqueeze(0), pe_i.unsqueeze(0)
        self.pe_r, self.pe_i = pe_r.to(device=x_r.device, dtype=x_r.dtype), pe_i.to(device=x_i.device, dtype=x_i.dtype)

    def forward(self, x_r, x_i):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, C, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, C, time, ...)

        """
        self.extend_pe(x_r, x_i)
        x_r, x_i = x_r * self.xscale + self.pe_r[:, : x_r.size(1), : x_r.size(2)], x_i * self.xscale + self.pe_i[:, : x_i.size(1), : x_r.size(2)]
        return self.dropout(x_r, x_i)


class ComplexConv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = ComplexSequential(
            ComplexConv2d(1, odim, 3, 2),
            ComplexLeakyReLU(),
            ComplexConv2d(odim, odim, 3, 2),
            ComplexLeakyReLU(),
        )
        self.out = ComplexSequential(
            ComplexLinear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )
        self.down_sample_list = [[3, 2], [3,2]]


    def forward(self, x_r, x_i, ilens=None):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """


        for kernel_size, stride in self.down_sample_list:
            max_ilen = torch.tensor(float(max(ilens)))
            for i in range(ilens.shape[0]):
                ilens[i] = torch.ceil(ilens[i].float()/stride) if (max_ilen-ilens[i] > kernel_size-1) else torch.ceil((max_ilen.float()-kernel_size+1)/stride)

        # x_r, x_i = x_r.unsqueeze(1), x_i.unsqueeze(1)  # (b, c, t, f)
        x_r, x_i = self.conv(x_r, x_i)
        b, c, t, f = x_r.size()
        x_r, x_i = self.out(
            x_r.transpose(1, 2).contiguous().view(b, t, c * f),
            x_i.transpose(1, 2).contiguous().view(b, t, c * f)
            )
        if ilens is None:
            return x_r, x_i, None
        return x_r, x_i, ilens

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class ComplexConv2dSubsampling6(nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = ComplexSequential(
            ComplexConv2d(1, odim, 3, 2),
            ComplexLeakyReLU(),
            ComplexConv2d(odim, odim, 5, 3),
            ComplexLeakyReLU(),
        )
        self.out = ComplexSequential(
            ComplexLinear(odim * (((idim - 1) // 2 - 2) // 3), odim),
        )
        self.down_sample_list = [[3, 2], [5, 3]]

    def forward(self, x_r, x_i, ilens=None):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """

        for kernel_size, stride in self.down_sample_list:
            max_ilen = torch.tensor(float(max(ilens)))
            for i in range(ilens.shape[0]):
                ilens[i] = torch.ceil(ilens[i].float()/stride) if (max_ilen-ilens[i] > kernel_size-1) else torch.ceil((max_ilen.float()-kernel_size+1)/stride)

        # x_r, x_i = x_r.unsqueeze(1), x_i.unsqueeze(1)  # (b, c, t, f)
        x_r, x_i = self.conv(x_r, x_i)
        b, c, t, f = x_r.size()
        x_r, x_i = self.out(
            x_r.transpose(1, 2).contiguous().view(b, t, c * f),
            x_i.transpose(1, 2).contiguous().view(b, t, c * f)
            )
        if ilens is None:
            return x_r, x_i, None
        return x_r, x_i, ilens

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class ComplexConv2dSubsampling8(nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param torch.nn.Module pos_enc: custom position encoding layer

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = ComplexSequential(
            ComplexConv2d(1, odim, 3, 2),
            ComplexLeakyReLU(),
            ComplexConv2d(odim, odim, 3, 2),
            ComplexLeakyReLU(),
            ComplexConv2d(odim, odim, 3, 2),
            ComplexLeakyReLU(),
        )
        self.out = ComplexSequential(
            ComplexLinear(odim * ((((idim - 1) // 2 - 1) // 2 -1)) // 2, odim),
        )
        self.down_sample_list = [[3, 2], [3, 2], [3, 2]]

    def forward(self, x_r, x_i, ilens=None):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]

        """
        for kernel_size, stride in self.down_sample_list:
            max_ilen = torch.tensor(float(max(ilens)))
            for i in range(ilens.shape[0]):
                ilens[i] = torch.ceil(ilens[i].float()/stride) if (max_ilen-ilens[i] > kernel_size-1) else torch.ceil((max_ilen.float()-kernel_size+1)/stride)

        # x_r, x_i = x_r.unsqueeze(1), x_i.unsqueeze(1)  # (b, c, t, f)
        x_r, x_i = self.conv(x_r, x_i)
        b, c, t, f = x_r.size()
        x_r, x_i = self.out(
            x_r.transpose(1, 2).contiguous().view(b, t, c * f),
            x_i.transpose(1, 2).contiguous().view(b, t, c * f)
            )
        if ilens is None:
            return x_r, x_i, None
        return x_r, x_i, ilens

    def __getitem__(self, key):
        """Subsample x.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class ComplexMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int in_feat: the number of in features
    :param int out_feat: the number of out features
    :param float dropout_rate: dropout rate

    """
    def __init__(self, n_head, in_feat, out_feat, dropout_rate):
        super().__init__()
        assert in_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = in_feat // n_head
        self.h = n_head
        self.linear_q = ComplexLinear(in_feat, in_feat)
        self.linear_k = ComplexLinear(in_feat, in_feat)
        self.linear_v = ComplexLinear(in_feat, in_feat)
        self.linear_out = ComplexLinear(in_feat, out_feat)
        self.softmax = ComplexSoftmax(dim=-1)
        self.attn = None
        self.dropout = ComplexDropout(p=dropout_rate)
        self.leaky_relu = ComplexLeakyReLU()
        self.reciprocal_sqrtdk = 1 / math.sqrt(self.d_k)
    
    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        :param torch.Tensor query: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor key: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor value: [(batch, channel, time, size),(batch, channel, time, size)]
        :return torch.Tensor transformed query, key and value (batch, head, time, channel, d_k)

        """
        n_batch = query[0].size(0)
        n_channel = query[0].size(1)
        q = [q.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for q in self.linear_q(*query)]  # (batch, head, time, channel, d_k)
        del query
        k = [k.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for k in self.linear_k(*key)]    # (batch, head, time, channel, d_k)
        del key
        v = [v.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for v in self.linear_v(*value)]  # (batch, head, time, channel, d_k)
        # v = self.leaky_relu(*v)

        return q, k, v

    def _masked_fill(self, scores, mask, min_value):
        return [scores[0].masked_fill(mask, min_value), scores[1].masked_fill(mask, min_value)]

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        :param torch.Tensor value: [(batch, head, time, channel, d_k),(batch, head, time, channel, d_k)]
        :param torch.Tensor scores: [(batch, head, time, channel, channel),(batch, head, time, channel, channel)]
        :param torch.Tensor mask: (batch, 1, time2) or (batch, time1, time2)
        :return torch.Tensor transformed `value` (batch, channel, time, d_model)
            weighted by the attention score (batch, channel, channel)

        """
        n_batch = value[0].size(0)
        n_channel = value[0].size(-2)
        if mask is not None:
            # (batch, 1, time2) or (batch, time1, time2) -> (batch, 1, 1, 1, time2) or (batch, 1, 1, time1, time2)
            mask = mask.unsqueeze(1).unsqueeze(1).eq(0)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores[0].dtype).numpy().dtype).min
            )
            # scores = self._masked_fill(scores, mask, min_value)
            scores_r, scores_i = self.softmax(*scores, mask, min_value)
            del scores
            self.attn = self._masked_fill([scores_r, scores_i], mask, 0.0)  # (batch, head, channel, time1, time2)
        else:
            scores_r, scores_i = self.softmax(*scores)  # (batch, head, time1, time2)
            del scores
            self.attn = [scores_r, scores_i]

        p_attn = self.dropout(*self.attn)
        
        x = self.matmul(p_attn, value)  # (batch, head, time, channel, d_k)
        del p_attn, value
        x = (
            x[0].transpose(1, 3).contiguous().view(n_batch, n_channel, -1, self.h * self.d_k),
            x[1].transpose(1, 3).contiguous().view(n_batch, n_channel, -1, self.h * self.d_k)
        )  # (batch, channel, time, d_model)
        
        # x = self.leaky_relu(*x)
        return self.linear_out(*x)  # (batch, channel, time, d_model)

    def conjugate_transpose_matmul_conjugate(self, m1, m2):
        # (m1 x m2^H)*
        m2 = [m2[0].transpose(-2, -1), -m2[1].transpose(-2, -1)]
        return [torch.matmul(m1[0],m2[0]) - torch.matmul(m1[1],m2[1]), -torch.matmul(m1[1],m2[0]) - torch.matmul(m1[0],m2[1])]

    def matmul(self, m1, m2):
        return [torch.matmul(m1[0],m2[0]) - torch.matmul(m1[1],m2[1]), torch.matmul(m1[1],m2[0]) + torch.matmul(m1[0],m2[1])]

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor key: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor value: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor mask: (batch, 1, time2) or (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output (batch, time1, d_model), (batch, time1, d_model)
        """

        q, k, v = self.forward_qkv(query, key, value)
        del query, key, value
        scores = self.conjugate_transpose_matmul_conjugate(q, k)  # (batch, head, time, channel, channel)
        del q, k
        scores = [score * self.reciprocal_sqrtdk for score in scores]
        
        return self.forward_attention(v, scores, mask)


class ComplexChannelWiseMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of in features
    :param int n_feat: the number of out features
    :param float dropout_rate: dropout rate

    """
    def __init__(self, n_head, in_feat, out_feat, dropout_rate):
        super().__init__()
        assert in_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = in_feat // n_head
        self.h = n_head
        self.linear_q = ComplexLinear(in_feat, in_feat)
        self.linear_k = ComplexLinear(in_feat, in_feat)
        self.linear_v = ComplexLinear(in_feat, in_feat)
        self.linear_out = ComplexLinear(in_feat, out_feat)
        self.attn = None
        self.dropout = ComplexDropout(p=dropout_rate)
        self.leaky_relu = ComplexLeakyReLU()
    
    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        :param torch.Tensor query: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor key: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor value: [(batch, channel, time, size),(batch, channel, time, size)]
        :return torch.Tensor transformed query, key and value

        """
        n_batch = query[0].size(0)
        n_channel = query[0].size(1)
        q = [q.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for q in self.linear_q(*query)]  # (batch, head, time, channel, d_k)
        k = [k.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for k in self.linear_k(*key)]    # (batch, head, time, channel, d_k)
        v = [v.contiguous().view(n_batch, n_channel, -1, self.h, self.d_k).transpose(1, 3) for v in self.linear_v(*value)]  # (batch, head, time, channel, d_k)
        v = self.leaky_relu(*v)

        return q, k, v

    def forward_attention(self, value, scores):
        """Compute attention context vector.

        :param torch.Tensor value: [(batch, head, time, channel, d_k),(batch, head, time, channel, d_k)]
        :param torch.Tensor scores: [(batch, head, time, channel, channel),(batch, head, time, channel, channel)]
        :return torch.Tensor transformed `value` (batch, channel, time, d_model)
            weighted by the attention score (batch, channel, channel)

        """
        n_batch = value[0].size(0)
        n_channel = value[0].size(-2)
        self.attn = scores

        p_attn = self.dropout(*self.attn)
        
        x = self.matmul(p_attn, value)  # (batch, head, time, channel, d_k)
        x = (
            x[0].transpose(1, 3).contiguous().view(n_batch, n_channel, -1, self.h * self.d_k),
            x[1].transpose(1, 3).contiguous().view(n_batch, n_channel, -1, self.h * self.d_k)
        )  # (batch, channel, time, d_model)
        
        x = self.leaky_relu(*x)
        return self.linear_out(*x)  # (batch, channel, time, d_model)

    def conjugate_transpose_matmul_conjugate(self, m1, m2):
        m2 = [m2[0].transpose(-2, -1), -m2[1].transpose(-2, -1)] # conj-transpose
        return [torch.matmul(m1[0],m2[0]) - torch.matmul(m1[1],m2[1]), -torch.matmul(m1[1],m2[0]) - torch.matmul(m1[0],m2[1])] # conj

    def matmul(self, m1, m2):
        return [torch.matmul(m1[0],m2[0]) - torch.matmul(m1[1],m2[1]), torch.matmul(m1[1],m2[0]) + torch.matmul(m1[0],m2[1])]

    def forward(self, query, key, value):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor key: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.Tensor value: [(batch, channel, time, size),(batch, channel, time, size)]
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output (batch, time1, d_model), (batch, time1, d_model)
        """

        q, k, v = self.forward_qkv(query, key, value)
        scores = self.conjugate_transpose_matmul_conjugate(q, k)  # (batch, head, time, channel, channel)
        scores = [score/math.sqrt(self.d_k) for score in scores]
        
        return self.forward_attention(v, scores)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


class ComplexSoftmax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        if not isinstance(dim, int):
            raise
        self.dim = dim
    
    def forward(self, x_r, x_i, mask=None, min_value=None):
        amplitude = torch.pow(x_r.pow(2) + x_i.pow(2), 0.5)
        if isinstance(mask, torch.Tensor):
            amplitude = amplitude.masked_fill(mask, min_value)
        
        scale = F.softmax(amplitude, dim=self.dim) / amplitude
        return scale * x_r, scale * x_i


class ComplexGELU(nn.Module):
    
    def __init__(self):
        super(ComplexGELU, self).__init__()
    def forward(self, x_r, x_i):
        return gelu_accurate(x_r), gelu_accurate(x_i)

class ComplexTransposeLast(Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x_r, x_i):
        if self.deconstruct_idx is not None:
            x_r = x_r[self.deconstruct_idx]
            x_i = x_i[self.deconstruct_idx]
        return x_r.transpose(-2, -1), x_i.transpose(-2, -1)



class ComplexSequential(Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t

class ComplexDropout(Module):
    def __init__(self,p=0.5, inplace=True):
        super(ComplexDropout,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        if self.p == 0.:
            return input_r, input_i
        return complex_dropout(input_r,input_i,self.p,self.inplace)

class ComplexDropout2d(Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout2d,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_dropout2d(input_r,input_i,self.p,self.inplace)

class ComplexMaxPool2d(Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,input_r,input_i):
        return complex_max_pool2d(input_r,input_i,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)
class ComplexAvgPool2d(Module):

    def __init__(self,kernel_size, stride=None, padding=0,
                        ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self,input_r,input_i):
        return complex_avg_pool2d(input_r,input_i,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                ceil_mode = self.ceil_mode, count_include_pad = self.count_include_pad,
                                )


class ComplexReLU(Module):

    def forward(self,input_r,input_i):
        return complex_relu(input_r,input_i)

class ComplexLeakyReLU(Module):
    def __init__(self, negative_slope: int = 0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_leaky_relu(input_r,input_i, self.negative_slope, self.inplace)

class ComplexConvTranspose2d(Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)


    def forward(self,input_r,input_i):
        return self.conv_tran_r(input_r)-self.conv_tran_i(input_i), \
               self.conv_tran_r(input_i)+self.conv_tran_i(input_r)

class ComplexConv1d(Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.r_weight = self.conv_r.weight
        self.i_weight = self.conv_i.weight

    def forward(self,input_r, input_i):
#        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)

class ComplexConv2d(Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True, transpose:List=None):
        super(ComplexConv2d, self).__init__()
        self.transpose = transpose
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self,input_r, input_i):

        if isinstance(self.transpose, list):
            input_r_t, input_i_t = input_r.transpose(*self.transpose), input_i.transpose(*self.transpose)
            return (self.conv_r(input_r_t)-self.conv_i(input_i_t)).transpose(*self.transpose), \
               (self.conv_r(input_i_t)+self.conv_i(input_r_t)).transpose(*self.transpose)
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), \
               self.fc_r(input_i)+self.fc_i(input_r)

class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)

class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])


            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r-mean_r[None, :, None, None]
            input_i = input_i-mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

            input_r = input_r-mean[None,:,0,None,None]
            input_i = input_i-mean[None,:,1,None,None]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
                           Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 2)
        #self._check_input_dim(input)

        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input_r = input_r-mean_r[None, :]
            input_i = input_i-mean_i[None, :]


            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = input_r.var(dim=0,unbiased=False)+self.eps
            Cii = input_i.var(dim=0,unbiased=False)+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=0)

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]
            # zero mean values
            input_r = input_r-mean[None,:,0]
            input_i = input_i-mean[None,:,1]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:]*input_r+Rri[None,:]*input_i, \
                           Rii[None,:]*input_i+Rri[None,:]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0]*input_r+self.weight[None,:,2]*input_i+\
                               self.bias[None,:,0], \
                               self.weight[None,:,2]*input_r+self.weight[None,:,1]*input_i+\
                               self.bias[None,:,1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input_r, input_i

class _ComplexGroupNorm(Module):

    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3)) # GroupNorm also has learnable weights per channel
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))  # num_channels * 2
            self.register_buffer('running_covar', torch.zeros(num_features,3))  # num_channels * 3
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)


class ComplexFp32GroupNorm(_ComplexGroupNorm):
    # Guo: Only suitable for N x C x L 
    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        assert(self.num_groups.__class__.__name__=="int"), "num_groups must be an integer, but found {}".format(self.num_groups.__class__.__name__)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        if self.training:

            # calculate mean of real and imaginary part
            N, C, L = input_r.shape
            shape_len = len(input_r.shape)
            channel_per_group = int(self.num_features/self.num_groups)
            
            n = channel_per_group * L
            C_mat = torch.zeros([self.num_features, self.num_features])
            for i in range(self.num_groups):
                C_mat[i*channel_per_group:(i+1)*channel_per_group,i*channel_per_group:(i+1)*channel_per_group] = torch.ones([channel_per_group, channel_per_group])

            mean_r = torch.matmul(C_mat, input_r.sum(dim=2)) / n
            mean_i = torch.matmul(C_mat, input_i.sum(dim=2)) / n
            
            mean = torch.stack((mean_r,mean_i),dim=2)  # N * num_channels x 2

            # update running mean
            with torch.no_grad():
                for i in range(N):
                    self.running_mean = exponential_average_factor * mean[i,:,:]\
                        + (1 - exponential_average_factor) * self.running_mean   # num_channels x 2

            input_r = input_r-mean_r[:, :, None]
            input_i = input_i-mean_i[:, :, None]

            # Elements of the covariance matrix (biased for train)
            # n = input_r.numel() / input_r.size(0)

            # N x C
            Crr = 1. / n * torch.matmul(C_mat, input_r.pow(2).sum(dim=2))+self.eps
            Cii = 1. / n * torch.matmul(C_mat, input_i.pow(2).sum(dim=2))+self.eps
            Cri = 1. / n * torch.matmul(c_mat, torch.mul(input_r,input_i).sum(dim=2))

            with torch.no_grad():
                for i in range(N):
                    self.running_covar[:,0] = exponential_average_factor * Crr[i,:] * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_covar[:,0]

                    self.running_covar[:,1] = exponential_average_factor * Cii[i,:] * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_covar[:,1]

                    self.running_covar[:,2] = exponential_average_factor * Cri[i,:] * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[None,:,0]+self.eps
            Cii = self.running_covar[None,:,1]+self.eps
            Cri = self.running_covar[None,:,2]#+self.eps

            input_r = input_r-mean[None,:,0].unsqueeze(-1)
            input_i = input_i-mean[None,:,1].unsqueeze(-1)

        # calculate the inverse square root the covariance matrix (NxC)
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[:,:,None]*input_r+Rri[:,:,None]*input_i, \
                           Rii[:,:,None]*input_i+Rri[:,:,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None]*input_r+self.weight[None,:,2,None]*input_i+\
                               self.bias[None,:,0,None], \
                               self.weight[None,:,2,None]*input_r+self.weight[None,:,1,None]*input_i+\
                               self.bias[None,:,1,None]

        return input_r, input_i

##########----------------------BAK for Layernorm
# class _ComplexLayerNorm(Module):
#     #   Input : N x L x C
#     def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(_ComplexLayerNorm, self).__init__()
#         self.num_groups = num_groups
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features,3)) # GroupNorm also has learnable weights per channel
#             self.bias = Parameter(torch.Tensor(num_features,2))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(2))  # N * num_channels * 2
#             self.register_buffer('running_covar', torch.zeros(3))
#             self.running_covar[0] = 1.4142135623730951
#             self.running_covar[1] = 1.4142135623730951
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_covar', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_covar.zero_()
#             self.running_covar[0] = 1.4142135623730951
#             self.running_covar[1] = 1.4142135623730951
#             self.num_batches_tracked.zero_()

#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             init.constant_(self.weight[:,:2],1.4142135623730951)
#             init.zeros_(self.weight[:,2])
#             init.zeros_(self.bias)


# class ComplexFp32LayerNorm(_ComplexLayerNorm):
#     # Guo: Only suitable for N x L x C (Channel in the last dimension) 
#     def forward(self, input_r, input_i):
#         assert(input_r.size() == input_i.size())
#         assert(len(input_r.shape) == 4)
#         # assert(self.num_groups.__class__.__name__=="int"), "num_groups must be an integer, but found {}".format(self.num_groups.__class__.__name__)
#         exponential_average_factor = 0.0


#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum


#         if self.training:

#             # calculate mean of real and imaginary part
#             N, C, L = input_r.shape
#             shape_len = len(input_r.shape)
#             # channel_per_group = int(self.num_features/self.num_groups)

#             mean_r = input_r.mean(dim=2)  # N * L
#             mean_i = input_i.mean(dim=2)
            
#             mean = torch.stack((mean_r,mean_i),dim=2)  # N * L x 2

#             # update running mean
#             with torch.no_grad():
#                 for i in range(N):
#                     for j in range(L):
#                         self.running_mean = exponential_average_factor * mean[i,j,:]\
#                             + (1 - exponential_average_factor) * self.running_mean   # 2

#             input_r = input_r-mean_r[:, :, None]
#             input_i = input_i-mean_i[:, :, None]

#             # Elements of the covariance matrix (biased for train)
#             # n = input_r.numel() / input_r.size(0)

#             # N x L
#             n = C
#             Crr = input_r.var(dim=2, unbiased=False) +self.eps
#             Cii = input_i.var(dim=2, unbiased=False) +self.eps
#             Cri = 1. / C * torch.mul(input_r,input_i).sum(dim=2)

#             # running_covar 1 x 1 x 3
#             with torch.no_grad():
#                 for i in range(N):
#                     for j in range(L):
#                         self.running_covar[0] = exponential_average_factor * Crr[i,j] * n / (n - 1)\
#                             + (1 - exponential_average_factor) * self.running_covar[0]

#                         self.running_covar[1] = exponential_average_factor * Cii[i,j] * n / (n - 1)\
#                             + (1 - exponential_average_factor) * self.running_covar[1]

#                         self.running_covar[2] = exponential_average_factor * Cri[i,j] * n / (n - 1)\
#                             + (1 - exponential_average_factor) * self.running_covar[2]

#         else:
#             mean = self.running_mean
#             Crr = self.running_covar[None,None,0]+self.eps
#             Cii = self.running_covar[None,None,1]+self.eps
#             Cri = self.running_covar[None,None,2]#+self.eps

#             input_r = input_r-mean[None,None,0,None]
#             input_i = input_i-mean[None,None,1,None]

#         # calculate the inverse square root the covariance matrix (NxL)
#         det = Crr*Cii-Cri.pow(2)
#         s = torch.sqrt(det)
#         t = torch.sqrt(Cii+Crr + 2 * s)
#         inverse_st = 1.0 / (s * t)
#         Rrr = (Cii + s) * inverse_st
#         Rii = (Crr + s) * inverse_st
#         Rri = -Cri * inverse_st

#         input_r, input_i = Rrr[:,:,None]*input_r+Rri[:,:,None]*input_i, \
#                            Rii[:,:,None]*input_i+Rri[:,:,None]*input_r

#         if self.affine:
#             input_r, input_i = self.weight[None,None,:,0]*input_r+self.weight[None,None,:,2]*input_i+\
#                                self.bias[None,None,:,0], \
#                                self.weight[None,None,:,2]*input_r+self.weight[None,None,:,1]*input_i+\
#                                self.bias[None,None,:,1]

#         return input_r, input_i

class _ComplexLayerNorm(Module):
    #   Input : N x T x F
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(_ComplexLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3)) # LayerNorm also has learnable weights per channel
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)


class ComplexFp32LayerNorm(_ComplexLayerNorm):
    # Guo: Only suitable for N x T x F (Channel in the last dimension) 
    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 3)
        # assert(self.num_groups.__class__.__name__=="int"), "num_groups must be an integer, but found {}".format(self.num_groups.__class__.__name__)
  

        B, T, F = input_r.shape
        mean_dim = [1,2]
        mean_r, mean_i = input_r.mean(dim=mean_dim), input_i.mean(dim=mean_dim)

        input_r = input_r-mean_r[:, None, None]
        input_i = input_i-mean_i[:, None, None]

        Crr = input_r.pow(2).mean(dim=mean_dim) + self.eps
        Cii = input_i.pow(2).mean(dim=mean_dim) + self.eps
        Cri = (input_r*input_i).mean(dim=mean_dim)
        # calculate the inverse square root the covariance matrix (B)
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[:,None,None]*input_r+Rri[:,None,None]*input_i, \
                           Rii[:,None,None]*input_i+Rri[:,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None, None, :, 0]*input_r+self.weight[None, None, :, 2]*input_i+\
                               self.bias[None,None,:,0], \
                               self.weight[None,None,:,2]*input_r+self.weight[None,None,:,1]*input_i+\
                               self.bias[None,None,:,1]

        return input_r, input_i

class _ComplexLayerNorm4(Module):
    #   Input : B x F x C x T 
    def __init__(self, num_channel, eps=1e-5, affine=True):
        super(_ComplexLayerNorm4, self).__init__()
        # self.num_groups = num_groups
        self.num_features = num_channel
        self.eps = eps
        # self.momentum = momentum
        self.affine = affine
        # self.track_running_stats = track_running_stats
        if self.affine:
            F = num_channel
            self.weight = Parameter(torch.Tensor(F, 3)) # LayerNorm also has learnable weights per channel
            self.bias = Parameter(torch.Tensor(F, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        # if self.track_running_stats:
        #     self.register_buffer('running_mean', torch.zeros(2))  # N * num_channels * 2
        #     self.register_buffer('running_covar', torch.zeros(3))
        #     self.running_covar[0] = 1.4142135623730951
        #     self.running_covar[1] = 1.4142135623730951
        #     self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        # else:
        #     self.register_parameter('running_mean', None)
        #     self.register_parameter('running_covar', None)
        #     self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    # def reset_running_stats(self):
    #     if self.track_running_stats:
    #         self.running_mean.zero_()
    #         self.running_covar.zero_()
    #         self.running_covar[0] = 1.4142135623730951
    #         self.running_covar[1] = 1.4142135623730951
    #         self.num_batches_tracked.zero_()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[..., :2],1.4142135623730951)
            init.zeros_(self.weight[..., 2])
            init.zeros_(self.bias)


class ComplexLayerNorm(_ComplexLayerNorm4):
    ''' 
    Guo: Only suitable for BFCT and normalize on the last three dimension
    '''
    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        
        B, F, C, T = input_r.shape
        # shape_len = len(input_r.shape)
            # channel_per_group = int(self.num_features/self.num_groups)
        # dim_num_features = tuple(range(-1, -1-len(self.num_features), -1))
        dim_num_features = [1, 2, 3]
        mean_r = input_r.mean(dim=dim_num_features)  # B
        mean_i = input_i.mean(dim=dim_num_features)
           
        mean = torch.stack((mean_r,mean_i), dim=1)  # B x 2

            # update running mean
            # with torch.no_grad():
            #     for i in range(N):
            #         for j in range(L):
            #             self.running_mean = exponential_average_factor * mean[i,j,:]\
            #                 + (1 - exponential_average_factor) * self.running_mean   # 2

        input_r = input_r - mean_r[:, None, None, None] # B x 1 x 1 x 1
        input_i = input_i - mean_i[:, None, None, None]

            # Elements of the covariance matrix (biased for train)
            # n = input_r.numel() / input_r.size(0)

            # n = input_r.numel() / input_r.size(1)
            # Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            # Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            # Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])
            
        Crr = input_r.pow(2).mean(dim=dim_num_features) + self.eps
        Cii = input_i.pow(2).mean(dim=dim_num_features) + self.eps
        Cri = (input_r*input_i).mean(dim=dim_num_features)

        # calculate the inverse square root the covariance matrix (B)
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[:,None,None,None]*input_r+Rri[:,None,None,None]*input_i, \
                           Rii[:,None,None,None]*input_i+Rri[:,None,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, None, None, 0]*input_r+self.weight[None, :, None, None, 2]*input_i+\
                               self.bias[None, :, None, None, 0], \
                               self.weight[None, :, None, None, 2]*input_r+self.weight[None, :, None, None, 1]*input_i+\
                               self.bias[None, :, None, None, 1]

        return input_r, input_i

class ComplexFp42LayerNorm(_ComplexLayerNorm4):
    ''' 
    Guo: Only suitable for BTCF and normalize on the last three dimension
    '''
    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        
        B, T, C, F = input_r.shape
        # shape_len = len(input_r.shape)
            # channel_per_group = int(self.num_features/self.num_groups)
        # dim_num_features = tuple(range(-1, -1-len(self.num_features), -1))
        dim_num_features = [1, 2, 3]
        mean_r = input_r.mean(dim=dim_num_features)  # B
        mean_i = input_i.mean(dim=dim_num_features)
           
        mean = torch.stack((mean_r,mean_i), dim=1)  # B x 2

            # update running mean
            # with torch.no_grad():
            #     for i in range(N):
            #         for j in range(L):
            #             self.running_mean = exponential_average_factor * mean[i,j,:]\
            #                 + (1 - exponential_average_factor) * self.running_mean   # 2

        input_r = input_r - mean_r[:, None, None, None] # B x 1 x 1 x 1
        input_i = input_i - mean_i[:, None, None, None]

            # Elements of the covariance matrix (biased for train)
            # n = input_r.numel() / input_r.size(0)

            # n = input_r.numel() / input_r.size(1)
            # Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            # Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            # Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])
            
        Crr = input_r.pow(2).mean(dim=dim_num_features) + self.eps
        Cii = input_i.pow(2).mean(dim=dim_num_features) + self.eps
        Cri = (input_r*input_i).mean(dim=dim_num_features)

        # calculate the inverse square root the covariance matrix (B)
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[:,None,None,None]*input_r+Rri[:,None,None,None]*input_i, \
                           Rii[:,None,None,None]*input_i+Rri[:,None,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None, None, None, :, 0]*input_r+self.weight[None, None, None, :, 2]*input_i+\
                               self.bias[None, None, None, :, 0], \
                               self.weight[None, None, None, :, 2]*input_r+self.weight[None, None, None, :, 1]*input_i+\
                               self.bias[None, None, None, :, 1]

        return input_r, input_i
