import argparse
import itertools
import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from deep_cloost.losses import MSELoss
from deep_cloost.model import Encoder
from deep_cloost.train import initialize_biases

RENDER_GRID = 5
SAMPLE_GRID = 5

USE_CUDA = torch.cuda.is_available()
DEVICE = (torch.device('cuda') if USE_CUDA else torch.device('cpu'))


class OutputLayer(nn.Module):
    def __init__(self, num_options, zero=True):
        super().__init__()
        self.num_options = num_options
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_options, 3, padding=1),
        )
        # Don't interfere with the biases by default.
        if zero:
            self.layers[-1].weight.detach().zero_()
            self.layers[-1].bias.detach().zero_()

    def forward(self, x):
        new_shape = x.shape[:1] + (self.num_options,) + x.shape[1:]
        return self.layers(x).view(new_shape)


def main():
    args = arg_parser().parse_args()
    train_loader, test_loader = create_datasets(args.batch)
    model = create_or_load_model(args)

    add_stages(args, train_loader, test_loader, model)

    for i in itertools.count():
        save_renderings(args, test_loader, model)
        save_samples(args, model)
        print('[tune %d] initial test loss: %f' % (i, evaluate_model(test_loader, model)))
        tune_model(args, train_loader, model)


def add_stages(args, train_loader, test_loader, model):
    for i in range(model.num_stages, args.latents):
        if args.no_pretrain:
            zeros = torch.zeros((args.options,) + model.shape).to(DEVICE)
            model.add_stage(OutputLayer(args.options, zero=False).to(DEVICE),
                            nn.Parameter(zeros))
            continue
        stage = i + 1
        samples = gather_samples(train_loader, args.init_samples)
        biases = initialize_biases(model, samples, batch=args.batch)
        model.add_stage(OutputLayer(args.options).to(DEVICE), biases)
        print('[stage %d] initial test loss: %f' % (stage, evaluate_model(test_loader, model)))
        if stage != 1:
            tune_model(args, train_loader, model)
            print('[stage %d] final test loss: %f' % (stage, evaluate_model(test_loader, model)))

        save_checkpoint(args, model)
        save_renderings(args, test_loader, model)
        save_samples(args, model)


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
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = model.reconstruct(inputs)
        loss += inputs.shape[0] * model.loss_fn(outputs, inputs)
        count += inputs.shape[0]
    return loss / count


def tune_model(args, loader, model):
    optimizer = optim.Adam(model.parameters(), lr=args.tune_lr)
    last_loss = None
    for i in range(args.tune_epochs):
        losses = []
        aux_losses = []
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            main_loss, aux_loss = model.train_losses(inputs)
            loss = main_loss + aux_loss * args.tune_aux_coeff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(main_loss.item())
            aux_losses.append(aux_loss.item())

        new_loss = np.mean(losses)
        if last_loss is not None and new_loss > last_loss:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.tune_lr_step
        last_loss = new_loss

        print('[stage %d] * [epoch %d] train loss: %f (aux %f)' %
              (model.num_stages, i, new_loss, np.mean(aux_losses)))


def create_or_load_model(args):
    if os.path.exists(args.checkpoint):
        print('=> loading encoder model from checkpoint...')
        return load_checkpoint(args)
    else:
        print('=> creating new encoder model...')
        return Encoder((1, 28, 28), args.options, lambda x: x, MSELoss())


def load_checkpoint(args):
    state = torch.load(args.checkpoint, map_location='cpu')
    model = Encoder((1, 28, 28), args.options, lambda x: x, MSELoss(),
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


def save_renderings(args, loader, model):
    data = gather_samples(loader, RENDER_GRID ** 2)
    with torch.no_grad():
        recons = model.reconstruct(data)
    img = torch.cat([data, recons], dim=-1).view(-1, 28 * 2).cpu().numpy()
    img = np.clip(((img * 0.3081) + 0.1307), 0, 1)
    img = (img * 255).astype('uint8')
    grid = np.concatenate(img.reshape([RENDER_GRID, -1, 28 * 2]), axis=-1)
    Image.fromarray(grid).save('renderings.png')


def save_samples(args, model):
    latents = torch.randint(high=model.options, size=(SAMPLE_GRID**2, model.num_stages))
    with torch.no_grad():
        data = model.decode(latents.to(DEVICE))
    img = data.view(-1, 28).cpu().numpy()
    img = np.clip(((img * 0.3081) + 0.1307), 0, 1)
    img = (img * 255).astype('uint8')
    grid = np.concatenate(img.reshape([SAMPLE_GRID, -1, 28]), axis=-1)
    Image.fromarray(grid).save('samples.png')


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--latents', default=20, type=int)
    parser.add_argument('--options', default=8, type=int)
    parser.add_argument('--init-samples', default=10000, type=int)
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--checkpoint', default='mnist_model.pt', type=str)

    parser.add_argument('--tune-epochs', default=1, type=int)
    parser.add_argument('--tune-lr', default=0.001, type=float)
    parser.add_argument('--tune-lr-step', default=0.3, type=float)
    parser.add_argument('--tune-aux-coeff', default=0.01, type=float)

    return parser


if __name__ == '__main__':
    main()
