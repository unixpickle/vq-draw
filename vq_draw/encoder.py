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
              reconstructions: a tensor of parameters to
                predict the outputs.
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
