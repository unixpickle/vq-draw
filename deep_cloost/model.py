import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, shape, options, base, loss_fn, output_fn=None, num_layers=0):
        self.shape = shape
        self.options = options
        self.base = base
        self.loss_fn = loss_fn
        self.output_layers = []
        self.output_biases = []
        for i in range(num_layers):
            layer = output_fn()
            bias = nn.Parameter(torch.zeros((options,) + shape))
            self.output_layers.append(layer)
            self.output_biases.append(bias)
            self.add_module('output%d' % i, layer)
            self.add_module('bias%d' % i, bias)

    def forward(self, inputs):
        """
        Apply the encoder and track the corresponding
        reconstructions.

        Returns:
          A tuple (encodings, reconstructions)
        """
        current_outputs = torch.zeros_like(inputs)
        encodings = []
        for i, (layer, bias) in enumerate(zip(self.output_layers, self.output_biases)):
            base_out = self.base(current_outputs)
            layer_out = bias + layer(base_out)
            new_outputs = current_outputs[:, None] + layer_out
            losses = torch.stack([torch.stack([self.loss_fn(new_outputs[i, j], inputs[i])
                                               for j in range(new_outputs.shape[1])])
                                  for i in range(new_outputs.shape[0])])
            indices = torch.argmin(losses, dim=-1)
            encodings.append(indices)
            current_outputs = new_outputs[:, indices]
        return torch.stack(encodings, dim=-1), current_outputs

    def decode(self, codes):
        current_outputs = torch.zeros((codes.shape[0],) + self.shape, device=self.device)
        for i, (layer, bias) in enumerate(zip(self.output_layers, self.output_biases)):
            base_out = self.base(current_outputs)
            layer_out = bias + layer(base_out)
            new_outputs = current_outputs[:, None] + layer_out
            current_outputs = new_outputs[:, codes[i]]
        return current_outputs

    def reconstruct(self, inputs, batch=None):
        if batch is None:
            return self(inputs)[1]
        results = []
        for i in range(0, inputs.shape, batch):
            results.append(self(inputs[i:i+batch])[1])
        return torch.cat(results, dim=0)

    @property
    def device(self):
        return next(self.parameters()).device
