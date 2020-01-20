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
from deep_cloost.train import initialize_biases, gather_samples, evaluate_model

RENDER_GRID = 5
SAMPLE_GRID = 5
IMG_SIZE = 32

USE_CUDA = torch.cuda.is_available()
DEVICE = (torch.device('cuda') if USE_CUDA else torch.device('cpu'))


class BaseLayer(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_options*3, 3, padding=1),
        )

    def forward(self, x):
        x = self.layers(x)
        new_shape = (x.shape[0], self.num_options, 3, *x.shape[2:])
        return x.view(new_shape)


class OutputLayer(nn.Module):
    def forward(self, x):
        return x


def main():
    args = arg_parser().parse_args()
    train_loader, test_loader = create_datasets(args.batch)
    model = create_or_load_model(args)

    add_stages(args, train_loader, test_loader, model)

    args.tune_epochs = 1
    for i in itertools.count():
        save_renderings(args, test_loader, model)
        save_samples(args, model)
        test_loss = evaluate_model(test_loader, model)
        train_loss = tune_model(args, train_loader, model, log=False)
        save_checkpoint(args, model)
        print('[tune %d] train=%f test=%f' % (i, train_loss, test_loss))


def add_stages(args, train_loader, test_loader, model):
    for i in range(model.num_stages, args.latents):
        if args.no_pretrain:
            model.add_stage(OutputLayer().to(DEVICE))
            continue
        stage = i + 1
        samples = gather_samples(train_loader, args.init_samples).to(DEVICE)
        biases = initialize_biases(model, samples, batch=args.batch)
        model.add_stage(OutputLayer().to(DEVICE), bias=biases)
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./cifar_data', train=True, download=True,
                         transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./cifar_data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def tune_model(args, loader, model, log=True):
    optimizer = optim.Adam(model.parameters(), lr=args.tune_lr)
    last_loss = None
    for i in range(args.tune_epochs):
        losses = []
        aux_losses = []
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            main_loss, all_loss, aux_loss = model.train_losses(inputs)
            loss = main_loss + all_loss + aux_loss * args.tune_aux_coeff
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

        if log:
            print('[stage %d] * [epoch %d] train loss: %f (aux %f)' %
                  (model.num_stages, i, new_loss, np.mean(aux_losses)))

    return last_loss


def create_or_load_model(args):
    if os.path.exists(args.checkpoint):
        print('=> loading encoder model from checkpoint...')
        return load_checkpoint(args)
    else:
        print('=> creating new encoder model...')
        return Encoder((3, IMG_SIZE, IMG_SIZE), args.options,
                       BaseLayer(args.options).to(DEVICE),
                       MSELoss())


def load_checkpoint(args):
    state = torch.load(args.checkpoint, map_location='cpu')
    model = Encoder((3, IMG_SIZE, IMG_SIZE), args.options, BaseLayer(args.options), MSELoss(),
                    output_fn=OutputLayer,
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
    data = gather_samples(loader, RENDER_GRID ** 2).to(DEVICE)
    with torch.no_grad():
        recons = model.reconstruct(data)
    img = torch.cat([data, recons], dim=-1).view(RENDER_GRID**2, 3, IMG_SIZE, IMG_SIZE * 2)
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    img = np.clip(((img * 0.5) + 0.5), 0, 1)
    img = (img * 255).astype('uint8')
    grid = np.concatenate(img.reshape([RENDER_GRID, -1, IMG_SIZE * 2, 3]), axis=-2)
    Image.fromarray(grid).save('renderings.png')


def save_samples(args, model):
    latents = torch.randint(high=model.options, size=(SAMPLE_GRID**2, model.num_stages))
    with torch.no_grad():
        data = model.decode(latents.to(DEVICE))
    img = data.view(-1, 3, IMG_SIZE, IMG_SIZE)
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    img = np.clip(((img * 0.5) + 0.5), 0, 1)
    img = (img * 255).astype('uint8')
    grid = np.concatenate(img.reshape([SAMPLE_GRID, -1, IMG_SIZE, 3]), axis=-2)
    Image.fromarray(grid).save('samples.png')


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', default=32, type=int)
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
