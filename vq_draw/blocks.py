from abc import abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn


class CondBlock(nn.Module):
    """
    Base class for blocks which take the stage index as
    one of the inputs. These blocks are conditioned on the
    stage, hence "cond".
    """
    @abstractmethod
    def forward(self, x, stage):
        pass


class Sequential(CondBlock, nn.Sequential):
    """
    A sequential block that passes the stage to other
    staged blocks.
    """

    def forward(self, x, stage):
        for b in self:
            if isinstance(b, CondBlock):
                x = b(x, stage)
            else:
                x = b(x)
        return x


class CondModule(CondBlock):
    """
    An arbitrary stage-conditioned module that encompasses
    multiple instances of the same module.
    """

    def __init__(self, num_stages, ctor):
        super().__init__()
        self.module_list = nn.ModuleList([ctor() for _ in range(num_stages)])

    def forward(self, x, stage):
        return self.module_list[stage](x)


class CondChannelMask(CondBlock):
    """
    A module which multiplies the channels by a
    stage-conditional vector.
    """

    def __init__(self, num_stages, channels, ones_init=False):
        super().__init__()
        if ones_init:
            self.embeddings = nn.Parameter(torch.ones(num_stages, channels))
        else:
            self.embeddings = nn.Parameter(torch.randn(num_stages, channels))

    def forward(self, x, stage):
        scale = self.embeddings[None, stage]
        while len(scale.shape) < len(x.shape):
            scale = scale[..., None]
        return x * scale


class ResidualBlock(Sequential):
    """
    A sequential module that adds its outputs to its
    inputs.
    """

    def forward(self, x, stage):
        return super().forward(x, stage) + x


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim, randomize=False):
        super().__init__()

        if randomize:
            self.pos_enc = nn.Parameter(torch.randn(1, dim, seq_len))
            return

        indices = np.arange(seq_len)[None, None].astype('float32')
        coeffs = np.zeros([1, dim, 1], dtype='float32')

        min_freq = 1 / (10 * seq_len)
        max_freq = 1
        for i in range(0, dim-1, 2):
            frac = i / dim
            freq = frac * max_freq + (1 - frac) * min_freq
            coeffs[0, i] = freq
        coeffs *= math.pi * 2

        pos_enc = torch.from_numpy(coeffs * indices).float()
        self.pos_enc = nn.Parameter(pos_enc)

    def forward(self, x):
        return x + self.pos_enc


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)
