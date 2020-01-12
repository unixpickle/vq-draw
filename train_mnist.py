import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from deep_cloost.model import Encoder
from deep_cloost.train import initialize_biases

USE_CUDA = torch.cuda.is_available()
DEVICE = (torch.device('cuda') if USE_CUDA else torch.device('cpu'))


class OutputLayer(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 28 * 28 * num_options),
        )
        # Don't interfere with the biases by default.
        self.layers[-1].weight.detach().zero_()
        self.layers[-1].bias.detach().zero_()

    def forward(self, x):
        new_shape = x.shape[:1] + (self.num_options,) + x.shape[1:]
        return self.layers(x.view(x.shape[0], -1)).view(new_shape)


def main():
    args = arg_parser().parse_args()
    train_loader, test_loader = create_datasets(args.batch)
    model = create_or_load_model(args)

    for i in range(model.num_stages, args.latents):
        print('Creating encoder stage %d...' % i)
        samples = gather_samples(train_loader, args.init_samples)
        biases = initialize_biases(model, samples, batch=args.batch)
        model.add_stage(OutputLayer(args.options), biases)
        print('Evaluating new encoder...')
        evaluate_model(test_loader, model)

        save_checkpoint(args, model)


def create_datasets(batch_size):
    # Taken from pytorch MNIST demo.
    kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def gather_samples(loader, num_samples):
    results = []
    count = 0
    while count < num_samples:
        for inputs, _ in loader:
            results.append(inputs)
            count += inputs.shape[0]
            if count >= num_samples:
                break
    return torch.cat(results, dim=0)[:num_samples].to(DEVICE)


def evaluate_model(loader, model):
    loss = 0.0
    count = 0
    for inputs, _ in loader:
        outputs = model.reconstruct(inputs)
        loss += inputs.shape[0] * model.loss_fn(outputs, inputs)
        count += inputs.shape[0]
    print('Mean evaluation loss: %f' % (loss/count,))


def create_or_load_model(args):
    if os.path.exists(args.checkpoint):
        print('Loading encoder model from checkpoint...')
        return load_checkpoint(args)
    else:
        print('Creating new encoder model...')
        return Encoder((1, 28, 28), args.options, lambda x: x, nn.MSELoss())


def load_checkpoint(args):
    state = torch.load(args.checkpoint, map_location='cpu')
    model = Encoder((1, 28, 28), args.options, lambda x: x, nn.MSELoss(),
                    output_fn=lambda: OutputLayer(args.options),
                    num_stages=state['num_stages'])
    model.load_state_dict(state['encoder'])
    return model.to(DEVICE)


def save_checkpoint(args, model):
    checkpoint = {
        'num_stages': model.num_stages,
        'encoder': model.state_dict(),
    }
    torch.save(checkpoint, args.checkpoint)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--latents', default=20, type=int)
    parser.add_argument('--options', default=8, type=int)
    parser.add_argument('--init-samples', default=10000, type=int)
    parser.add_argument('--checkpoint', default='mnist_model.pt', type=str)
    return parser


if __name__ == '__main__':
    main()