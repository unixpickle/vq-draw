from abc import abstractmethod

import torch.nn as nn


class SegmentRefiner(nn.Module):
    """
    A refiner which splits up a refinement sequences into
    segments that are handled by different models.
    """

    def __init__(self, seg_len, *segments):
        super().__init__()
        self.seg_len = seg_len
        self.segments = nn.ModuleList(segments)

    def forward(self, x, stage):
        seg = self.segments[stage // self.seg_len]
        return seg(x, stage % self.seg_len)


class ResidualRefiner(nn.Module):
    """
    Base class for refiner modules that compute additive
    residuals for the inputs.
    """
    @abstractmethod
    def residuals(self, x, stage):
        """
        Generate a set of potential deltas to the input.
        """
        pass

    def forward(self, x, stage):
        return x[:, None] + self.residuals(x, stage)
