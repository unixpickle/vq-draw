from abc import abstractmethod

import torch
import torch.nn as nn


class LossFunc(nn.Module):
    """
    A loss function that can be applied in a 2-D batch.
    """

    @abstractmethod
    def loss_grid(self, output, target):
        """
        Compute the losses for a target batch, given a set
        of potential predictions.

        Args:
            output: an [N x C x ...] Tensor.
            target: an [N x 1 x ...] Tensor.

        Returns:
            An [N x C] Tensor of losses.
        """
        pass


class MSELoss(LossFunc):
    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2))

    def loss_grid(self, x, y):
        return torch.mean(torch.pow(x - y, 2), dim=tuple(range(2, len(x.shape))))


class SoftmaxLoss(LossFunc):
    def forward(self, x, y):
        log_probs = torch.log_softmax(x, dim=-1)
        return -torch.mean(torch.gather(log_probs, -1, y[..., None]))

    def loss_grid(self, x, y):
        y = y.repeat(1, x.shape[1], *([1] * (len(y.shape) - 2)))
        log_probs = torch.log_softmax(x, dim=-1)
        losses = -torch.gather(log_probs, -1, y[..., None])
        return torch.mean(losses, dim=tuple(range(2, len(x.shape))))
