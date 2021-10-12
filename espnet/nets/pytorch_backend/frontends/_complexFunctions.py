#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

from torch.nn.functional import relu, max_pool2d, dropout, dropout2d, pad, avg_pool2d, leaky_relu

def complex_relu(input_r,input_i):
    return relu(input_r), relu(input_i)

def complex_leaky_relu(input_r,input_i, negative_slope, inplace=False):
    return leaky_relu(input_r, negative_slope=negative_slope), leaky_relu(input_i, negative_slope=negative_slope, inplace=inplace)

def complex_max_pool2d(input_r,input_i,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):

    return max_pool2d(input_r, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices), \
           max_pool2d(input_i, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices)

def complex_avg_pool2d(input_r,input_i, kernel_size, stride=None, padding=0,
                                ceil_mode=False, count_include_pad=True):

    return avg_pool2d(input_r, kernel_size, stride, padding,
                      ceil_mode, count_include_pad), \
           avg_pool2d(input_i, kernel_size, stride, padding,
                      ceil_mode, count_include_pad)

def complex_dropout(input_r,input_i, p=0.5, training=True, inplace=False):
    return dropout(input_r, p, training, inplace), \
           dropout(input_i, p, training, inplace)


def complex_dropout2d(input_r,input_i, p=0.5, training=True, inplace=False):
    return dropout2d(input_r, p, training, inplace), \
           dropout2d(input_i, p, training, inplace)

def complex_zero_pad(input_r, input_i, pad_tensor):
    return pad(input_r, pad_tensor, "constant", 0), pad(input_i, pad_tensor, "constant", 0)
