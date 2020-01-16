import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def loss_grid(self, x, y):
        return torch.mean(torch.pow(x - y, 2), dim=tuple(range(2, len(x.shape))))

    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2))
