from abc import abstractmethod

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

    def __init__(self, num_stages, channels):
        super().__init__()
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
