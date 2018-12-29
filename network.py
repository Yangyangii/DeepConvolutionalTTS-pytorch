from config import ConfigArgs as args
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as norm
import numpy as np
import module as mm

class TextEncoder(nn.Module):
    """
    Text Encoder
        T: (N, Cx, Tx) Text embedding (variable length)
    Returns:
        K: (N, Cx, Tx) Text Encoding for Key
        V: (N, Cx, Tx) Text Encoding for Value
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.Conv1d(args.Ce, args.Cx*2, 1, padding='same', activation_fn=torch.relu))])  # filter up to split into K, V
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cx*2, args.Cx*2, 1, padding='same', activation_fn=None))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 3, dilation=3**i, padding='same'))
                               for _ in range(2) for i in range(4)])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 3, dilation=1, padding='same'))
                               for i in range(2)])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 1, dilation=1, padding='same'))
                               for i in range(2)])

    def forward(self, L):
        y = L
        for i in range(len(self.hc_blocks)):
            y = self.hc_blocks[i](y)
        K, V = y.chunk(2, dim=1)  # half size for axis Cx
        return K, V

class AudioEncoder(nn.Module):
    """
    Text Encoder
        prev_audio: (N, n_mels, Ty/r) Mel-spectrogram (variable length)
    Returns:
        Q: (N, Cx, Ty/r) Audio Encoding for Query
    """

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.CausalConv1d(args.n_mels, args.Cx, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cx, args.Cx, 1, activation_fn=torch.relu))
                               for _ in range(2)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cx, args.Cx, 3, dilation=3**i)) # i is in [[0,1,2,3],[0,1,2,3]]
                               for _ in range(2) for i in range(4)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cx, args.Cx, 3, dilation=3))
                               for i in range(2)])
        # self.hc_blocks.extend([mm.CausalConv1d(args.Cy, args.Cx, 1, dilation=1, activation_fn=torch.relu)]) # down #filters to dotproduct K, V

    def forward(self, S):
        Q = S
        for i in range(len(self.hc_blocks)):
            Q = self.hc_blocks[i](Q)
        return Q

class DotProductAttention(nn.Module):
    """
    Dot Product Attention
    Args:
        K: (N, Cx, Tx)
        V: (N, Cx, Tx)
        Q: (N, Cx, Ty)
    Returns:
        R: (N, Cx, Ty)
        A: (N, Tx, Ty) alignments
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, K, V, Q):
        A = torch.softmax((torch.bmm(K.transpose(1, 2), Q)/np.sqrt(args.Cx)), dim=1) # K.T.dot(Q) -> (N, Tx, Ty)
        R = torch.bmm(V, A) # (N, Cx, Ty)
        return R, A

class AudioDecoder(nn.Module):
    """
    Dot Product Attention
    Args:
        R_: (N, Cx*2, Ty)
    Returns:
        O: (N, n_mels, Ty)
    """
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.CausalConv1d(args.Cx*2, args.Cy, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cy, args.Cy, 3, dilation=3**i))
                               for i in range(4)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cy, args.Cy, 3, dilation=1))
                               for _ in range(2)])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cy, args.Cy, 1, dilation=1, activation_fn=torch.relu))
                               for _ in range(3)])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cy, args.n_mels, 1, dilation=1))]) # down #filters to dotproduct K, V

    def forward(self, R_):
        Y = R_
        for i in range(len(self.hc_blocks)):
            Y = self.hc_blocks[i](Y)
        return torch.sigmoid(Y)
