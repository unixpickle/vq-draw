from abc import ABC, abstractmethod, abstractproperty
import argparse
import os

from PIL import Image
import numpy as np
import torch
import torch.optim as optim


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


class Trainer(ABC):
    """
    A Trainer can be used for end-to-end training of an
    encoder model from a CLI application.

    Simply override all of the abstract methods and
    properties to specify your model and dataset.
    """

    def __init__(self):
        self.args = self.arg_parser().parse_args()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        self.train_loader, self.test_loader = self.create_datasets()
        self.model = self.create_or_load_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def arg_parser(self):
        """
        Create an argument parser for CLI arguments.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--batch', default=32, type=int)
        parser.add_argument('--stages', default=self.default_stages, type=int)
        parser.add_argument('--options', default=8, type=int)
        parser.add_argument('--checkpoint', default=self.default_checkpoint, type=str)
        parser.add_argument('--save-interval', default=10, type=int)

        parser.add_argument('--grad-checkpoint', action='store_true')
        parser.add_argument('--grad-decay', default=0, type=float)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--aux-coeff', default=0.01, type=float)
        parser.add_argument('--final-coeff', default=1, type=float)

        return parser

    def main(self):
        """
        Run the infinite training loop.
        """
        loaders = zip(self._cycle_batches(self.train_loader),
                      self._cycle_batches(self.test_loader))
        for i, (train_batch, test_batch) in enumerate(loaders):
            losses = self.model.train_losses(train_batch, checkpoint=self.args.grad_checkpoint)
            with torch.no_grad():
                test_losses = self.model.train_losses(test_batch,
                                                      checkpoint=self.args.grad_checkpoint)
            loss = (losses['choice'] +
                    losses['final'] * self.args.final_coeff +
                    losses['all'] * self.args.aux_coeff)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('step %d: train=%f test=%f entropy=%f' %
                  (i, losses['final'].item(), test_losses['final'].item(), losses['entropy']))
            if not i % self.args.save_interval:
                self.save_checkpoint()
                self.save_renderings()
                self.save_samples()

    def create_or_load_model(self):
        """
        Create the model and load it from a file if there
        is a saved checkpoint.
        """
        model = self.create_model()
        if os.path.exists(self.args.checkpoint):
            print('=> loading encoder model from checkpoint...')
            model.load_state_dict(torch.load(self.args.checkpoint, map_location='cpu'))
        else:
            print('=> created new encoder model...')
        model.num_stages = self.args.stages
        model.grad_decay = self.args.grad_decay
        return model.to(self.device)

    def save_checkpoint(self):
        """
        Save a checkpoint of the current model.
        """
        torch.save(self.model.state_dict(), self.args.checkpoint)

    def save_renderings(self):
        """
        Save a reconstruction grid to a file.
        """
        data = gather_samples(self.test_loader, self.image_grid_size ** 2).to(self.device)
        with torch.no_grad():
            recons = self.model.reconstruct(data)
        img = torch.cat([data, recons], dim=-1)
        self.save_grid('renderings.png', img)

    def save_samples(self):
        """
        Save a sample grid to a file.
        """
        latents = torch.randint(high=self.model.options,
                                size=(self.image_grid_size**2, self.model.num_stages))
        with torch.no_grad():
            img = self.model.decode(latents.to(self.device))
        self.save_grid('samples.png', img)

    def save_grid(self, path, images):
        """
        Save an arbitrary grid of images, where the images
        are of batch size self.image_grid_size**2, and
        have all but the last dimension in common with
        self.shape.
        """
        img = images.view(self.image_grid_size**2, *self.shape[:-1], -1)
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        img = np.clip(self.denormalize_image(img), 0, 1)
        img = (img * 255).astype('uint8')
        new_shape = [self.image_grid_size, self.shape[1]*self.image_grid_size, -1, self.shape[0]]
        grid = np.concatenate(img.reshape(new_shape), axis=-2).squeeze()
        Image.fromarray(grid).save(path)

    @property
    def image_grid_size(self):
        """
        Determine the number of images to save.
        """
        return 5

    @abstractmethod
    def denormalize_image(self, img):
        """
        Turn an image from the model/dataset into a Tensor
        of floats from 0 to 1.
        """
        pass

    @abstractproperty
    def default_checkpoint(self):
        """
        Get the default checkpoint name for the CLI.
        """
        pass

    @abstractproperty
    def default_stages(self):
        """
        Get the default number of stages for the CLI.
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Get the shape of the encoded images.
        """
        pass

    @abstractmethod
    def create_datasets(self):
        """
        Create the (train, test) data loaders.
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Create a new encoder model.
        """
        pass

    def _cycle_batches(self, loader):
        """
        Utility to infinitely cycle through batches from a
        data loader.
        """
        while True:
            for batch, _ in loader:
                yield batch.to(self.device)
