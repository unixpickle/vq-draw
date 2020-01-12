from scipi.cluster.vq import kmeans
import torch
import torch.nn as nn


def initialize_biases(encoder, data, batch=None):
    with torch.no_grad():
        recon = encoder.reconstruct(data, batch=batch)
    recon = recon.requires_grad_(True)
    loss = encoder.loss_fn(recon, data)
    grads, = torch.autograd.grad(loss, recon)
    clusters, _ = kmeans(grads.cpu().numpy(), encoder.options)
    return nn.Parameter(torch.from_numpy(clusters).to(encoder.device))
