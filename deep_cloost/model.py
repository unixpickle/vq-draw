import math

import torch
import torch.nn as nn
import torch.utils.checkpoint


class Encoder(nn.Module):
    def __init__(self, shape, options, base, loss_fn, output_fn=None, num_stages=0):
        super().__init__()
        self.shape = shape
        self.options = options
        self.base = base
        self.loss_fn = loss_fn
        self.output_layers = []
        self.output_biases = []
        for _ in range(num_stages):
            layer = output_fn()
            bias = nn.Parameter(torch.zeros((options,) + shape))
            self.add_stage(layer, bias=bias)

    def forward(self, inputs, checkpoint=False):
        """
        Apply the encoder and track the corresponding
        reconstructions.

        Args:
            inputs: a Tensor of inputs to encode.
            checkpoint: if True, use sqrt(stages) memory
              for longer reconstruction sequences.

        Returns:
            A tuple (encodings, reconstructions, losses):
              encodings: an [N x num_stages] tensor.
              reconstructions: a tensor like inputs.
              losses: an [N x num_stages x options] tensor.
        """
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
            return torch.zeros((inputs.shape[0], 0), dtype=torch.long), current_outputs
        return (torch.stack(encodings, dim=-1),
                current_outputs,
                torch.stack(all_losses, dim=1))

    def decode(self, codes):
        current_outputs = torch.zeros((codes.shape[0],) + self.shape, device=codes.device)
        for i in range(self.num_stages):
            new_outputs = self.apply_stage(i, current_outputs)
            current_outputs = new_outputs[range(new_outputs.shape[0]), codes[:, i]]
        return current_outputs

    def reconstruct(self, inputs, batch=None):
        if batch is None:
            return self(inputs)[1]
        results = []
        for i in range(0, inputs.shape[0], batch):
            results.append(self(inputs[i:i+batch])[1])
        return torch.cat(results, dim=0)

    def train_losses(self, inputs, **kwargs):
        losses = self(inputs, **kwargs)[-1]
        return (torch.mean(torch.min(losses[:, -1], dim=-1)[0]),
                torch.mean(torch.min(losses, dim=-1)[0]),
                torch.mean(losses))

    def apply_stage(self, idx, x):
        layer = self.output_layers[idx]
        bias = self.output_biases[idx]
        base_out = self.base(x)
        layer_out = layer(base_out)
        layer_out = bias + layer_out
        return x[:, None] + layer_out

    def add_stage(self, layer, bias=None):
        i = self.num_stages
        self.output_layers.append(layer)
        self.add_module('output%d' % i, layer)

        if bias is None:
            device = next(self.parameters()).device
            zeros = torch.zeros((self.options,) + self.shape).to(device)
            bias = nn.Parameter(zeros)

        self.output_biases.append(bias)
        self.register_parameter('bias%d' % i, bias)

    @property
    def num_stages(self):
        return len(self.output_layers)


class CIFARBaseLayer(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.output_scale = nn.Parameter(torch.tensor(0.05))
        self.layers = nn.Sequential(
            SkipConnect(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.GroupNorm(8, 64),
                SkipConnect(
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, 128),
                    SkipConnect(
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.GroupNorm(8, 256),
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.GroupNorm(8, 256),
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.GroupNorm(8, 128),
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.GroupNorm(8, 128),
                    ),
                    nn.ConvTranspose2d(128+128, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, 128),
                ),
                nn.ConvTranspose2d(64+128, 128, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.GroupNorm(8, 128),
            ),
            nn.Conv2d(3+128, num_options*3, 3, padding=1),
        )

    def forward(self, x):
        x = self.layers(x) * self.output_scale
        new_shape = (x.shape[0], self.num_options, 3, *x.shape[2:])
        return x.view(new_shape)


class SkipConnect(nn.Sequential):
    def forward(self, x):
        return torch.cat([super().forward(x), x], dim=1)
