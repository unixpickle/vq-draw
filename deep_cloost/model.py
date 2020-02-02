from abc import abstractmethod, abstractproperty
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint


class AbstractEncoder(nn.Module):
    def __init__(self, shape, options):
        super().__init__()
        self.shape = shape
        self.options = options

    @abstractmethod
    def forward(self, inputs, checkpoint=False, current_outputs=None):
        """
        Apply the encoder and track the corresponding
        reconstructions.

        Args:
            inputs: a Tensor of inputs to encode.
            checkpoint: if True, use sqrt(stages) memory
              for longer reconstruction sequences.
            current_outputs: if specified, this is a
              Tensor of the initial reconstructions.

        Returns:
            A tuple (encodings, reconstructions, losses):
              encodings: an [N x num_stages] tensor.
              reconstructions: a tensor like inputs.
              losses: an [N x num_stages x options] tensor.
        """
        pass

    @abstractmethod
    def decode(self, codes, current_outputs=None):
        pass

    @abstractproperty
    def num_stages(self):
        pass

    def reconstruct(self, inputs, **kwargs):
        return self(inputs)[1]

    def train_losses(self, inputs, **kwargs):
        losses = self(inputs, **kwargs)[-1]
        codes = torch.argmin(losses, dim=-1).view(-1)
        counts = torch.tensor([torch.sum(codes == i) for i in range(self.options)])
        probs = counts.float() / float(codes.shape[0])
        entropy = -torch.sum(torch.log(probs.clamp(1e-8, 1)) * probs)
        return {
            'final': torch.mean(torch.min(losses[:, -1], dim=-1)[0]),
            'choice': torch.mean(torch.min(losses, dim=-1)[0]),
            'all': torch.mean(losses),
            'entropy': entropy,
            'used': len(set(codes.cpu().numpy())),
        }


class SegmentedEncoder(AbstractEncoder):
    def __init__(self, encoders):
        super().__init__(encoders[0].shape, encoders[0].options)
        self.encoders = encoders
        for i, enc in enumerate(encoders):
            self.add_module('encoder%d' % i, enc)

    @property
    def num_stages(self):
        return sum(e.num_stages for e in self.encoders)

    def forward(self, inputs, checkpoint=False, current_outputs=None):
        encodings = []
        all_losses = []
        for enc in self.encoders:
            encs, current_outputs, losses = enc(inputs,
                                                checkpoint=checkpoint,
                                                current_outputs=current_outputs)
            encodings.append(encs)
            all_losses.append(losses)
        return (torch.cat(encodings, dim=-1),
                current_outputs,
                torch.cat(all_losses, dim=1))

    def decode(self, codes, current_outputs=None):
        if current_outputs is None:
            current_outputs = torch.zeros((codes.shape[0],) + self.shape, device=codes.device)
        start_idx = 0
        for enc in self.encoders:
            end_idx = start_idx + enc.num_stages
            current_outputs = enc.decode(codes[:, start_idx:end_idx],
                                         current_outputs=current_outputs)
            start_idx = end_idx
        return current_outputs


class Encoder(AbstractEncoder):
    def __init__(self, shape, options, refiner, loss_fn, num_stages=0, grad_decay=0):
        super().__init__(shape, options)
        self.refiner = refiner
        self.loss_fn = loss_fn
        self.bias = nn.Parameter(torch.zeros(options, *shape))
        self.register_buffer('_stages', torch.tensor(num_stages, dtype=torch.long))
        self.register_buffer('_grad_decay', torch.tensor(grad_decay, dtype=torch.float))

    @property
    def num_stages(self):
        return self._stages.item()

    @num_stages.setter
    def num_stages(self, x):
        self._stages.copy_(torch.zeros_like(self._stages) + x)

    @property
    def grad_decay(self):
        return self._grad_decay.item()

    @grad_decay.setter
    def grad_decay(self, x):
        self._grad_decay.copy_(torch.zeros_like(self._grad_decay) + x)

    def forward(self, inputs, checkpoint=False, current_outputs=None):
        if current_outputs is None:
            current_outputs = torch.zeros_like(inputs)
        interval = int(math.sqrt(self.num_stages))
        if not checkpoint or interval < 1:
            return self._forward_range(range(self.num_stages), inputs, current_outputs)
        encodings = []
        all_losses = []
        for i in range(0, self.num_stages, interval):
            r = range(i, min(i+interval, self.num_stages))

            def f(inputs, current_outputs, dummy, stages=r):
                encs, outs, losses = self._forward_range(stages, inputs, current_outputs)

                # Cannot have a return value that does not require
                # grad, so we use this hack instead.
                encodings.append(encs)

                return outs, losses

            # Workaround from:
            # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/16.
            dummy = torch.zeros(1).requires_grad_()
            current_outputs, losses = torch.utils.checkpoint.checkpoint(f, inputs, current_outputs,
                                                                        dummy)
            all_losses.append(losses)
        return (torch.cat(encodings, dim=-1),
                current_outputs,
                torch.cat(all_losses, dim=1))

    def _forward_range(self, stages, inputs, current_outputs):
        encodings = []
        all_losses = []
        for i in stages:
            new_outputs = self.apply_stage(i, current_outputs)
            losses = self.loss_fn.loss_grid(new_outputs, inputs[:, None])
            all_losses.append(losses)
            indices = torch.argmin(losses, dim=1)
            encodings.append(indices)
            current_outputs = new_outputs[range(new_outputs.shape[0]), indices]
        if len(encodings) == 0:
            return (torch.zeros(inputs.shape[0], 0, dtype=torch.long),
                    current_outputs,
                    torch.zeros(inputs.shape[0], 0, self.options))
        return (torch.stack(encodings, dim=-1),
                current_outputs,
                torch.stack(all_losses, dim=1))

    def decode(self, codes, current_outputs=None, num_stages=None):
        if num_stages is None:
            num_stages = self.num_stages
        if current_outputs is None:
            current_outputs = torch.zeros((codes.shape[0],) + self.shape, device=codes.device)
        for i in range(num_stages):
            new_outputs = self.apply_stage(i, current_outputs)
            current_outputs = new_outputs[range(new_outputs.shape[0]), codes[:, i]]
        return current_outputs

    def apply_stage(self, idx, x):
        x = x.detach() * self._grad_decay + x * (1 - self._grad_decay)
        res = self.refiner(x, idx)
        if idx == 0:
            res = res + self.bias
        return res


class ResidualRefiner(nn.Module):
    @abstractmethod
    def residuals(self, x, stage):
        """
        Generate a set of potential deltas to the input.
        """
        pass

    def forward(self, x, stage):
        return x[:, None] + self.residuals(x, stage)


class CIFARRefiner(ResidualRefiner):
    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.01))

        def res_block():
            return ResidualBlock(
                nn.ReLU(),
                nn.GroupNorm(8, 256),
                StagedConv2d(max_stages, 256, 256, 3, padding=1),
                nn.ReLU(),
                nn.GroupNorm(8, 256),
                StagedConv2d(max_stages, 256, 256, 3, padding=1),
            )

        self.layers = StagedSequential(
            # Reduce spatial resolution.
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            StagedConv2d(max_stages, 64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Process data at lower spatial resolution.
            StagedConv2d(max_stages, 128, 256, 3, padding=1),
            res_block(),
            res_block(),

            # Increase spacial resolution back to original.
            StagedConvTranspose2d(max_stages, 256, 128, 3, stride=2, padding=1,
                                  output_padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            StagedConvTranspose2d(max_stages, 128, 128, 3, stride=2, padding=1,
                                  output_padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            # Generate option outputs.
            nn.Conv2d(128, 3 * self.num_options, 3, padding=1),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
        return x.view(x.shape[0], self.num_options, 3, *x.shape[2:])


class MNISTRefiner(ResidualRefiner):
    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = StagedSequential(
            StagedConv2d(max_stages, 1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            StagedConv2d(max_stages, 32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            StagedConv2d(max_stages, 64, 128, 3, padding=1),
            nn.ReLU(),
            StagedConv2d(max_stages, 128, 128, 3, padding=1),
            nn.ReLU(),
            StagedConv2d(max_stages, 128, 64, 3, padding=1),
            nn.ReLU(),
            StagedConvTranspose2d(max_stages, 64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            StagedConvTranspose2d(max_stages, 64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_options, 3, padding=1),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
        return x.view(x.shape[0], self.num_options, 1, *x.shape[2:])


class StagedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, stage):
        pass


class StagedSequential(StagedBlock, nn.Sequential):
    def forward(self, x, stage):
        for b in self:
            if isinstance(b, StagedBlock):
                x = b(x, stage)
            else:
                x = b(x)
        return x


class StagedConv2d(StagedBlock):
    def __init__(self, num_options, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.embeddings = nn.Parameter(torch.randn(num_options, self.conv.out_channels))

    def forward(self, x, stage):
        return self.conv(x) * self.embeddings[stage, :, None, None]


class StagedConvTranspose2d(StagedBlock):
    def __init__(self, num_options, *args, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)
        self.embeddings = nn.Parameter(torch.randn(num_options, self.conv.out_channels))

    def forward(self, x, stage):
        return self.conv(x) * self.embeddings[stage, :, None, None]


class SkipConnect(StagedSequential):
    def forward(self, x, stage):
        return torch.cat([super().forward(x, stage), x], dim=1)


class ResidualBlock(StagedSequential):
    def forward(self, x, stage):
        return super().forward(x, stage) + x
