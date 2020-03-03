from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint


class Encoder(nn.Module):
    """
    The core encoder/decoder algorithm.

    This module uses a refinement network to iteratively
    generate better and better reconstructions of inputs,
    saving choices along the way as latent codes.

    Args:
        shape: the shape of the Tensors output by this
          model (not including the batch).
        options: the number of options at each stage,
          which must be matched by the refiner.
        refiner: a module which proposes refinements.
          Inputs are [N x *shape].
          Outputs are [N x options x *shape].
        loss_fn: a LossFunc to use as the refinement
          criterion.
        num_stages: initial number of stages.
        grad_decay: initial gradient decay.
    """

    def __init__(self, shape, options, refiner, loss_fn, num_stages=0, grad_decay=0):
        super().__init__()
        self.shape = shape
        self.options = options
        self.refiner = refiner
        self.loss_fn = loss_fn
        self.bias = nn.Parameter(torch.zeros(options, *shape))
        self.register_buffer('_stages', torch.tensor(num_stages, dtype=torch.long))
        self.register_buffer('_grad_decay', torch.tensor(grad_decay, dtype=torch.float))

    @property
    def num_stages(self):
        """
        Get the number of stages used for forward passes
        through the model.
        """
        return self._stages.item()

    @num_stages.setter
    def num_stages(self, x):
        self._stages.copy_(torch.zeros_like(self._stages) + x)

    @property
    def grad_decay(self):
        """
        Get a gradient decay factor, from 0 to 1, where 1
        means that no gradients flow back through stages.
        """
        return self._grad_decay.item()

    @grad_decay.setter
    def grad_decay(self, x):
        self._grad_decay.copy_(torch.zeros_like(self._grad_decay) + x)

    def forward(self, inputs, checkpoint=False, epsilon=0):
        """
        Apply the encoder and track the corresponding
        reconstructions.

        Args:
            inputs: a Tensor of inputs to encode.
            checkpoint: if True, use sqrt(stages) memory
              for longer reconstruction sequences.
            epsilon: probability of sampling randon latent
              codes.

        Returns:
            A tuple (encodings, reconstructions, losses):
              encodings: an [N x num_stages] tensor.
              reconstructions: a tensor like of parameters
                to predict the outputs.
              losses: an [N x num_stages x options] tensor.
        """
        current_outputs = torch.zeros((inputs.shape[0], *self.shape), device=inputs.device)
        interval = int(math.sqrt(self.num_stages))
        if not checkpoint or interval < 1:
            return self._forward_range(range(self.num_stages), inputs, current_outputs, epsilon)
        encodings = []
        all_losses = []
        for i in range(0, self.num_stages, interval):
            r = range(i, min(i+interval, self.num_stages))

            def f(inputs, current_outputs, dummy, stages=r):
                encs, outs, losses = self._forward_range(stages, inputs, current_outputs, epsilon)

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

    def _forward_range(self, stages, inputs, current_outputs, epsilon):
        encodings = []
        all_losses = []
        for i in stages:
            new_outputs = self.apply_stage(i, current_outputs)
            losses = self.loss_fn.loss_grid(new_outputs, inputs[:, None])
            all_losses.append(losses)
            indices = torch.argmin(losses, dim=1)
            if epsilon:
                rands = torch.randint(0, self.options, indices.shape, device=indices.device)
                takes = (torch.rand(indices.shape, device=indices.device) < epsilon).long()
                indices = takes * rands + (1 - takes) * indices
            encodings.append(indices)
            current_outputs = new_outputs[range(new_outputs.shape[0]), indices]
        if len(encodings) == 0:
            return (torch.zeros(inputs.shape[0], 0, dtype=torch.long, device=inputs.device),
                    current_outputs,
                    torch.zeros(inputs.shape[0], 0, self.options, device=inputs.device))
        return (torch.stack(encodings, dim=-1),
                current_outputs,
                torch.stack(all_losses, dim=1))

    def apply_stage(self, idx, x):
        """
        Apply a stage (number idx) to a batch of inputs x
        and get a batch of refinement proposals.
        """
        x = x.detach() * self._grad_decay + x * (1 - self._grad_decay)
        res = self.refiner(x, idx)
        if idx == 0:
            res = res + self.bias
        return res

    def reconstruct(self, inputs, **kwargs):
        """
        Reconstruct the inputs by encoding and decoding.
        """
        return self(inputs, **kwargs)[1]

    def train_quantities(self, inputs, **kwargs):
        """
        Get a dict of quantities which are useful for
        training, including losses and entropy.
        """
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

    def decode(self, codes, num_stages=None):
        """
        Create reconstructions for some latent codes.

        Args:
            codes: a long Tensor of shape [N x stages].
            num_stages: the number of decoding stages to
              run (useful for partial reconstructions).
        """
        if num_stages is None:
            num_stages = self.num_stages
        current_outputs = torch.zeros((codes.shape[0],) + self.shape, device=codes.device)
        for i in range(num_stages):
            new_outputs = self.apply_stage(i, current_outputs)
            current_outputs = new_outputs[range(new_outputs.shape[0]), codes[:, i]]
        return current_outputs


class SegmentRefiner(nn.Module):
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

    def __init__(self, num_options, max_stages):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.layers = Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
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
            nn.Conv2d(64, num_options, 3, padding=1),
        )

    def residuals(self, x, stage):
        x = self.layers(x, stage) * self.output_scale
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

        def res_block(dilation):
            return ResidualBlock(
                nn.ReLU(),
                nn.LayerNorm((128, seq_len)),
                nn.Conv1d(128, 128, 3, stride=1, padding=dilation, dilation=dilation),
                CondChannelMask(max_stages, 128),
                nn.ReLU(),
                nn.LayerNorm((128, seq_len)),
                nn.Conv1d(128, 512, 1),
                CondChannelMask(max_stages, 512),
                nn.ReLU(),
                nn.LayerNorm((512, seq_len)),
                nn.Conv1d(512, 128, 1),
                CondChannelMask(max_stages, 128),
            )

        self.embed = nn.Sequential(
            nn.Conv1d(vocab_size, 128, 1),
            nn.ReLU(),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, 128, seq_len))
        self.layers = Sequential(
            res_block(1),
            res_block(2),
            res_block(4),
            res_block(8),
            res_block(16),
            res_block(32),
            res_block(1),
            res_block(2),
            res_block(4),
            res_block(8),
            res_block(16),
            res_block(32),
            nn.ReLU(),
            nn.LayerNorm((128, seq_len)),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, num_options * vocab_size, 1),
        )

    def residuals(self, x, stage):
        # Use probabilities and scale to have a closer-to-
        # normal distribution.
        out = torch.softmax(x, dim=-1) * math.sqrt(x.shape[-1])
        out = out.permute(0, 2, 1)
        out = self.embed(out)
        out = out + self.pos_enc
        out = self.layers(out, stage)
        out = out.view(x.shape[0], self.num_options, self.vocab_size, self.seq_len)
        out = out.permute(0, 1, 3, 2).contiguous()
        return out * self.output_scale


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
