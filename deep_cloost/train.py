from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn


def initialize_biases(encoder, data, batch=None):
    """
    Use clustering to generate initial biases for the next
    stage of the encoder.

    Args:
        encoder: an Encoder model.
        data: a Tensor of shape [N x *encoder.shape].
        batch: the batch size to use for evaluating the
          data in the model.

    Returns:
        An nn.Parameter to use as the biases.
    """
    with torch.no_grad():
        recon = encoder.reconstruct(data, batch=batch)
    recon = recon.detach().requires_grad_(True)
    loss = encoder.loss_fn(recon, data)
    grads, = torch.autograd.grad(loss, recon)
    grads /= torch.std(grads)
    dense_grads = grads.view(grads.shape[0], -1)
    clusters, labels = kmeans2(dense_grads.cpu().numpy(), encoder.options, minit='++', iter=30)

    biases = -torch.from_numpy(clusters).to(data.device)
    biases = biases.view(-1, *grads.shape[1:])
    biases = _scale_line_search(recon, data, encoder.loss_fn, biases, labels)

    return nn.Parameter(biases)


def _scale_line_search(recon, data, loss, biases, labels):
    results = []
    for i in range(biases.shape[0]):
        indices = [j for j, x in enumerate(labels) if x == i]
        sub_recon = recon[indices]
        sub_data = data[indices]
        bias = biases[i]

        def loss_fn(alpha):
            new_recon = sub_recon + bias * alpha
            new_loss = loss(new_recon, sub_data)
            return new_loss.item()
        alpha = float(minimize_scalar(loss_fn, bracket=(0, 1000)).x)
        results.append(bias * alpha)
    return torch.stack(results)


def gather_samples(loader, num_samples):
    """
    Gather a batch of samples from a data loader.
    """
    results = []
    count = 0
    while count < num_samples:
        for inputs, _ in loader:
            results.append(inputs)
            count += inputs.shape[0]
            if count >= num_samples:
                break
    return torch.cat(results, dim=0)[:num_samples]


def evaluate_model(loader, model):
    """
    Evaluate the reconstruction loss for a model on an
    entire dataset, represented by a loader.

    Returns:
        A scalar Tensor of the loss.
    """
    device = torch.device('cpu')
    for p in model.parameters():
        device = p.device
        break

    loss = 0.0
    count = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.reconstruct(inputs)
            loss += inputs.shape[0] * model.loss_fn(outputs, inputs)
        count += inputs.shape[0]
    return loss / count
