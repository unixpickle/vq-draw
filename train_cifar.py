import argparse
import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from deep_cloost.losses import MSELoss
from deep_cloost.model import Encoder, CIFARRefiner
from deep_cloost.train import gather_samples

RENDER_GRID = 5
SAMPLE_GRID = 5
IMG_SIZE = 32

USE_CUDA = torch.cuda.is_available()
DEVICE = (torch.device('cuda') if USE_CUDA else torch.device('cpu'))


def main():
    args = arg_parser().parse_args()
    train_loader, test_loader = create_datasets(args.batch)
    model = create_or_load_model(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loaders = zip(cycle_batches(train_loader), cycle_batches(test_loader))
    for i, (train_batch, test_batch) in enumerate(loaders):
        loss_final, loss_all, loss_aux = model.train_losses(train_batch,
                                                            checkpoint=args.grad_checkpoint)
        with torch.no_grad():
            loss_test, _, _ = model.train_losses(test_batch, checkpoint=args.grad_checkpoint)
        loss = loss_final + loss_all + loss_aux * args.aux_coeff
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('step %d: train=%f test=%f' % (i, loss_final.item(), loss_test.item()))
        if not i % args.save_interval:
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


def cycle_batches(loader):
    while True:
        for batch, _ in loader:
            yield batch.to(DEVICE)


def create_or_load_model(args):
    model = Encoder(shape=(3, IMG_SIZE, IMG_SIZE),
                    options=args.options,
                    refiner=CIFARRefiner(args.options),
                    loss_fn=MSELoss(),
                    num_stages=args.stages)
    if os.path.exists(args.checkpoint):
        print('=> loading encoder model from checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        model.num_stages = args.stages
    else:
        print('=> created new encoder model...')
    return model.to(DEVICE)


def save_checkpoint(args, model):
    torch.save(model.state_dict(), args.checkpoint)


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
    parser.add_argument('--stages', default=20, type=int)
    parser.add_argument('--options', default=8, type=int)
    parser.add_argument('--checkpoint', default='mnist_model.pt', type=str)
    parser.add_argument('--save-interval', default=10, type=int)

    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--aux-coeff', default=0.01, type=float)

    return parser


if __name__ == '__main__':
    main()
