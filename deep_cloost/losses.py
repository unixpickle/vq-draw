import torch


class MSELoss:
    def loss_grid(self, x, y):
        return torch.mean(torch.pow(x - y, 2), dim=tuple(range(2, len(x.shape))))

    def loss(self, x, y):
        return torch.mean(torch.pow(x - y, 2))
