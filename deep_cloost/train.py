from scipi.cluster.vq import kmeans2
import torch
import torch.nn as nn


def initialize_biases(encoder, data, batch=None):
    with torch.no_grad():
        recon = encoder.reconstruct(data, batch=batch)
    recon = recon.requires_grad_(True)
    loss = encoder.loss_fn(recon, data)
    grads, = torch.autograd.grad(loss, recon)
    dense_grads = grads.view(grads.shape[0], -1)
    clusters, _ = kmeans2(dense_grads.cpu().numpy(), encoder.options, minit='++')
    torch_clusters = torch.from_numpy(clusters).to(encoder.device)
    # TODO: line search to find optimal scales for each
    # cluster center.
    return nn.Parameter(torch_clusters.view(-1, *grads.shape[1:]))
