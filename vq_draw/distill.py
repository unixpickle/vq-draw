from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            A tuple (latents, reconstructions):
              latents: an [N x stages] long Tensor of
                latents.
              reconstructions: an [N x ...] Tensor which
                represents the reconstructions.
        """
        latents = self.encode(inputs)
        return latents, self.decode(latents)

    @abstractmethod
    def encode_nll(self, inputs, latents):
        """
        Compute the negative log likelihood of the given
        latents.

        Args:
            inputs: a Tensor of inputs to encode, shape
              is [N x ...].
            latents: an [N x stages] long Tensor of
              latents.

        Returns:
            A floating point Tensor measuring the NLL of
              the latents, averaged across the batch and
              stages.
        """
        pass

    @abstractmethod
    def encode(self, inputs, sample=False):
        """
        Compute the latents for some inputs.

        Args:
            inputs: a Tensor of inputs to encode, shape
              is [N x ...].
            sample: if True, sample uniformly from the
              latent codes. Otherwise, attempt to select
              the most likely code.

        Returns:
            An [N x stages] long Tensor of latent codes.
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
        )
        self.dist = RNNDist(128, stages, options)

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

    def encode_nll(self, x, latents):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder_final(x)
        return self.dist.nll(x, latents)

    def encode(self, x, sample=False):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder_final(x)
        return self.dist.discretize(x, sample=sample)

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


class IndependentDist(nn.Module):
    def __init__(self, dim, stages, options):
        super().__init__()
        self.stages = stages
        self.options = options
        self.layer = nn.Linear(dim, stages * options)

    def nll(self, inputs, latents):
        latent_pred = self.layer(inputs)
        latent_pred = latent_pred.view(-1, self.stages, self.options)
        latent_pred = latent_pred.permute(0, 2, 1)
        return F.nll_loss(F.log_softmax(latent_pred, dim=1), latents)

    def discretize(self, inputs, sample=False):
        latent_pred = self.layer(inputs)
        latent_pred = latent_pred.view(-1, self.stages, self.options)
        probs = F.softmax(latent_pred, dim=-1)
        if not sample:
            return torch.argmax(probs, dim=-1)
        thresholds = torch.rand_like(probs[..., :1])
        samples = torch.sum((torch.cumsum(probs, dim=-1) < thresholds).long(), dim=-1)
        return samples


class RNNDist(nn.Module):
    def __init__(self, dim, stages, options):
        super().__init__()
        self.stages = stages
        self.options = options
        self.rnn = nn.GRU(dim, dim, 2)
        self.embedding = nn.Embedding(options, dim)
        self.init_state = nn.Parameter(torch.randn(self.rnn.num_layers, 1, dim))
        self.output_layer = nn.Linear(dim, options)

    def nll(self, inputs, latents):
        # [N x T x C]
        embedded = self.embedding(latents)
        embedded = torch.cat([torch.zeros_like(embedded[:, :1]), embedded[:, :-1]], dim=1)

        # [T x N x C]
        input_seq = (embedded + inputs[:, None]).permute(1, 0, 2)

        hidden = self.init_state.repeat(1, inputs.shape[0], 1)
        output_seq, _ = self.rnn(input_seq, hidden)

        # [N x T x C]
        output_seq = output_seq.permute(1, 0, 2)
        # [N x T x options]
        output_logits = self.output_layer(output_seq)
        output_logs = F.log_softmax(output_logits, dim=-1)
        # [N x options x T]
        output_logs = output_logs.permute(0, 2, 1)

        return F.nll_loss(output_logs, latents)

    def discretize(self, inputs, sample=False):
        latents = []
        hidden = self.init_state.repeat(1, inputs.shape[0], 1)
        next_inputs = inputs[None]
        for _ in range(self.stages):
            outputs, hidden = self.rnn(next_inputs, hidden)
            logits = self.output_layer(outputs[0])
            if sample:
                probs = F.softmax(logits, dim=-1)
                thresholds = torch.rand_like(probs[0, :1])
                samples = torch.sum((torch.cumsum(probs, dim=-1) < thresholds).long(), dim=-1)
            else:
                samples = torch.argmax(logits, dim=-1)
            embedded = self.embedding(samples)
            next_inputs = (inputs + embedded)[None]
            latents.append(samples)
        return torch.stack(latents, dim=1)
