import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .blocks import (ResidualBlock, CondChannelMask, Sequential, CondModule,
                     PositionalEncoding, TransformerEncoderLayer)
from .refiner import ResidualRefiner


class CIFARRefiner(ResidualRefiner):
    """
    A refiner module appropriate for the CIFAR dataset.
    """

    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        def res_block():
            return ResidualBlock(
                nn.ReLU(),
                nn.GroupNorm(8, 256),
                nn.Conv2d(256, 256, 3, padding=1),
                CondChannelMask(max_stages, 256),
                nn.ReLU(),
                nn.GroupNorm(8, 256),
                nn.Conv2d(256, 1024, 1),
                CondChannelMask(max_stages, 1024),
                nn.ReLU(),
                nn.GroupNorm(32, 1024),
                nn.Conv2d(1024, 256, 1),
                CondChannelMask(max_stages, 256),
            )

        self.layers = Sequential(
            # Reduce spatial resolution.
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.Conv2d(128, 256, 3, padding=1),
            CondChannelMask(max_stages, 256),
            res_block(),
            res_block(),
            res_block(),
            res_block(),
            res_block(),
            res_block(),

            # Increase spacial resolution back to original.
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Generate option outputs.
            nn.Conv2d(128, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.Conv2d(128, 3 * self.num_options, 5, padding=2),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
        return x.view(x.shape[0], self.num_options, 3, *x.shape[2:])


class CelebARefiner(ResidualRefiner):
    """
    A refiner module appropriate for the CelebA dataset.
    """

    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        def res_block():
            return ResidualBlock(
                nn.ReLU(),
                nn.GroupNorm(8, 128),
                nn.Conv2d(128, 128, 3, padding=1),
                CondChannelMask(max_stages, 128),
                nn.ReLU(),
                nn.GroupNorm(8, 128),
                nn.Conv2d(128, 512, 1),
                CondChannelMask(max_stages, 512),
                nn.ReLU(),
                nn.GroupNorm(16, 512),
                nn.Conv2d(512, 128, 1),
                CondChannelMask(max_stages, 128),
            )

        self.layers = Sequential(
            # Reduce spatial resolution by 8x.
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(4, 64),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            res_block(),
            res_block(),
            res_block(),
            res_block(),
            res_block(),
            res_block(),

            # Increase spacial resolution back to original.
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Generate option outputs.
            nn.Conv2d(128, 3 * self.num_options, 5, padding=2),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
        return x.view(x.shape[0], self.num_options, 3, *x.shape[2:])


class MNISTRefiner(ResidualRefiner):
    """
    A refiner module appropriate for the MNIST dataset.
    """

    def __init__(self, num_options, max_stages, gaussian=False):
        super().__init__()
        self.num_options = num_options
        self.gaussian = gaussian
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = Sequential(
            nn.Conv2d(1 if not gaussian else 2, 32, 3, stride=2, padding=1),
            CondChannelMask(max_stages, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            CondChannelMask(max_stages, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            CondChannelMask(max_stages, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 64),
            nn.ReLU(),
            nn.Conv2d(64, num_options if not gaussian else num_options*2, 3, padding=1),
        )

    def residuals(self, x, stage):
        if self.gaussian:
            means = x[..., 0]
            stds = torch.exp(x[..., 1])
            x = torch.cat([means, stds], dim=1)
        x = self.layers(x, stage) * self.output_scale
        if self.gaussian:
            x = x.view(x.shape[0], -1, 2, *x.shape[2:])
            x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x.view(x.shape[0], self.num_options, 1, *x.shape[2:])


class SVHNRefiner(ResidualRefiner):
    """
    A refiner module appropriate for the SVHN dataset.
    """

    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = Sequential(
            # Downsample the image to 8x8.
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            CondChannelMask(max_stages, 64),
            nn.ReLU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Process the downsampled image.
            nn.Conv2d(128, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 256, 3, padding=1),
            CondChannelMask(max_stages, 256),
            nn.ReLU(),
            nn.GroupNorm(8, 256),
            nn.Conv2d(256, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Upsample the image.
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # More powerful conditioning for output, which
            # gives better results.
            nn.Conv2d(128, 128, 3, padding=1),
            CondChannelMask(max_stages, 128),
            nn.ReLU(),
            CondModule(max_stages, lambda: nn.Conv2d(128, num_options * 3, 1)),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
        return x.view(x.shape[0], self.num_options, 3, *x.shape[2:])


class TextRefiner(ResidualRefiner):
    """
    A refiner module appropriate for textual dataset.
    """

    def __init__(self, num_options, max_stages, seq_len, vocab_size):
        super().__init__()
        self.num_options = num_options
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.output_scale = nn.Parameter(torch.tensor(0.1))

        def block():
            return Sequential(
                TransformerEncoderLayer(512, 8, dim_feedforward=2048, dropout=0),
                CondChannelMask(max_stages, 512, ones_init=True),
            )

        self.embed = nn.Sequential(
            nn.Conv1d(vocab_size, 512, 1),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(seq_len, 512)
        self.layers = Sequential(
            block(),
            block(),
            block(),
            block(),
            block(),
            block(),
            nn.Conv1d(512, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, num_options * vocab_size, 1),
        )

    def residuals(self, x, stage):
        # Use probabilities and scale to have a closer-to-
        # normal distribution.
        out = torch.softmax(x, dim=-1) * math.sqrt(x.shape[-1])
        out = out.permute(0, 2, 1)
        out = self.embed(out)
        out = self.pos_enc(out)
        out = self.layers(out, stage)
        out = out.view(x.shape[0], self.num_options, self.vocab_size, self.seq_len)
        out = out.permute(0, 1, 3, 2).contiguous()
        return out * self.output_scale
