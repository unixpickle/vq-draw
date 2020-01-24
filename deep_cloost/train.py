import torch


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
