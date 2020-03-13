from abc import abstractmethod
import math

import torch
import torch.nn as nn


class DistillAE(nn.Module):
    """
    Base class for distilled discrete auto-encoders.
    """

    def forward(self, inputs):
        """
        Produce latents and reconstructions for inputs.

        Args:
            inputs: a Tensor of inputs to encode, shape
              is [N x ...].

        Returns:
            A tuple (latent_logits, reconstructions):
              latent_logits: an [N x stages x options]
                Tensor of prediction logits.
              reconstructions: an [N x ...] Tensor which
                represents the reconstructions.
        """
        logits = self.encode(inputs)
        latents = torch.argmax(logits, dim=-1)
        return logits, self.decode(latents)

    @abstractmethod
    def encode(self, inputs):
        """
        Predict the latent codes for the inputs.

        Args:
            inputs: a Tensor of inputs to encode, shape
              is [N x ...].

        Returns:
            An [N x stages x options] Tensor of prediction
              logits.
        """
        pass

    @abstractmethod
    def decode(self, latents):
        """
        Decode a batch of Tensors from discrete latents.

        Args:
            latents: a Tensor of integers, of shape
              [N x stages].

        Returns:
            An [N x ...] Tensor of reconstructions.
        """

    def reconstruct(self, inputs):
        """
        Produce reconstructions for an input Tensor.
        """
        return self.forward(inputs)[-1]


class MNISTDistillAE(DistillAE):
    def __init__(self, stages, options):
        super().__init__()

        self.num_stages = stages
        self.options = options

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder_final = nn.Sequential(
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Linear(128, stages * options),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(stages * options, 128),
            nn.ReLU(),
            nn.Linear(128, 7*7*64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder_final(x)
        x = x.view(x.shape[0], self.num_stages, self.options)
        return x

    def decode(self, x):
        one_hot = torch.zeros(*x.shape, self.options, device=x.device)
        one_hot.scatter_(-1, x[..., None], 1)

        # Normalize for faster learning.
        one_hot = one_hot * math.sqrt(self.num_stages * self.options)

        x = one_hot.view(one_hot.shape[0], -1)
        x = self.decoder_input(x)
        x = x.view(x.shape[0], 64, 7, 7)
        x = self.decoder(x)
        return x
