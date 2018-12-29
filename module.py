from config import ConfigArgs as args
import torch
import torch.nn as nn
import numpy as np

class Conv1d(nn.Conv1d):
    """
    Hightway Convolution 1d
    Args:
        x: (N, C_in, L)
    Returns:
        y: (N, C_out, L)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(self.drop_rate) if self.drop_rate > 0 else None

    def forward(self, x):
        y = super(Conv1d, self).forward(x)
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        return y

class HighwayConv1d(Conv1d):
    """
    Hightway Convolution 1d
    Args:
        x: (N, C_in, T)
    Returns:
        y: (N, C_out, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True):
        self.drop_rate = drop_rate
        super(HighwayConv1d, self).__init__(in_channels, out_channels*2, kernel_size, activation_fn=None,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(self.drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        y = super(HighwayConv1d, self).forward(x) # (N, C_out*2, T)
        h, y_ = y.chunk(2, dim=1) # half size for axis C_out. (N, C_out, T) respectively
        h = torch.sigmoid(h) # Gate
        y_ = torch.relu(y_)
        y_ = h*y_ + (1-h)*x
        y_ = self.drop_out(y_) if self.drop_out is not None else y_
        return y_

class CausalConv1d(Conv1d):
    """
    Causal convolution 1d
    Args:
        x: (N, C_in, L)
    Returns:
        y: (N, C_out, L)
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, dilation=1, groups=1, bias=True):
        padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           activation_fn=activation_fn, drop_rate=drop_rate,
                                           stride=stride, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, x):
        y = super(CausalConv1d, self).forward(x)
        return y[:, :, :x.size(2)]  # (N, C, :-(ksize-1)) slicing

class CausalHighwayConv1d(CausalConv1d):
    """
    Causal convolution 1d
    Args:
        x: (N, C_in, L)
    Returns:
        y: (N, C_out, L)
    """
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.,
                 stride=1, dilation=1, groups=1, bias=True):
        self.drop_rate = drop_rate
        super(CausalHighwayConv1d, self).__init__(in_channels, out_channels*2, kernel_size,
                                           activation_fn=None,
                                           stride=stride, dilation=dilation,
                                                  groups=groups, bias=bias)
        self.drop_out = nn.Dropout(self.drop_rate) if self.drop_rate > 0 else None

    def forward(self, x):
        y = super(CausalHighwayConv1d, self).forward(x)
        h, y_ = y.chunk(2, dim=1)  # half size for axis C_out
        h = torch.sigmoid(h)  # Gate
        y_ = torch.relu(y_)
        y_ = h*y_ + (1.0-h)*x
        y_ = self.drop_out(y_) if self.drop_out is not None else y_
        return y_

class ConvTranspose1d(nn.ConvTranspose1d):
    """
    Transposed Convolution 1d
    Args:
        x: (N, C_in, L)
    Returns:
        y: (N, C_out, L*stride) naive shape
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(self.drop_rate) if self.drop_rate > 0 else None

    def forward(self, x):
        y = super(ConvTranspose1d, self).forward(x)
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        return y
