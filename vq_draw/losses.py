from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GaussianLoss(LossFunc):
    """
    A gaussian log-likelihood loss for inputs with a final
    dimension of 2 (for mean and log standard deviations).
    """

    def forward(self, x, y):
        return -torch.mean(self.log_probs(x, y))

    def loss_grid(self, x, y):
        return -torch.mean(self.log_probs(x, y), dim=tuple(range(2, len(y.shape))))

    def log_probs(self, x, y):
        mean = x[..., 0].contiguous()
        log_std = x[..., 1].contiguous()
        std = torch.exp(log_std)
        return -0.5 * (torch.pow((y - mean) / std, 2) + 2 * log_std + math.log(2 * math.pi))


class SoftmaxLoss(LossFunc):
    def forward(self, x, y):
        log_probs = torch.log_softmax(x, dim=-1)
        return -torch.mean(torch.gather(log_probs, -1, y[..., None]))

    def loss_grid(self, x, y):
        y = y.repeat(1, x.shape[1], *([1] * (len(y.shape) - 2)))
        log_probs = torch.log_softmax(x, dim=-1)
        losses = -torch.gather(log_probs, -1, y[..., None])
        return torch.mean(losses, dim=tuple(range(2, len(x.shape))))


class BCELoss(LossFunc):
    def forward(self, x, y):
        y_bcast = y + torch.zeros_like(x)
        return F.binary_cross_entropy_with_logits(x, y_bcast)

    def loss_grid(self, x, y):
        y_bcast = y + torch.zeros_like(x)
        losses = F.binary_cross_entropy_with_logits(x, y_bcast, reduction='none')
        return torch.mean(losses, dim=tuple(range(2, len(x.shape))))
