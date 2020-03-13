from abc import ABC, abstractmethod, abstractproperty
import argparse
import os
import random

from PIL import Image
import numpy as np
import torch
import torch.optim as optim


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
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr, betas=(0.9, 0.99), eps=1e-5)

    def arg_parser(self):
        """
        Create an argument parser for CLI arguments.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--batch', default=128, type=int)
        parser.add_argument('--stages', default=self.default_stages, type=int)
        parser.add_argument('--active-stages', default=0, type=int)
        parser.add_argument('--options', default=64, type=int)
        parser.add_argument('--checkpoint', default=self.default_checkpoint, type=str)
        parser.add_argument('--save-interval', default=10, type=int)
        parser.add_argument('--step-interval', default=1, type=int)
        parser.add_argument('--step-limit', default=0, type=int)

        if self.default_segment is not None:
            parser.add_argument('--segment', default=self.default_segment, type=int)

        parser.add_argument('--grad-checkpoint', action='store_true')
        parser.add_argument('--grad-decay', default=0, type=float)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--lr-final', default=0, type=float)
        parser.add_argument('--aux-coeff', default=0.01, type=float)
        parser.add_argument('--epsilon', default=0, type=float)
        parser.add_argument('--final-coeff', default=0, type=float)

        return parser

    def main(self):
        """
        Run the training loop.

        This may run forever, depending on the step limit
        CLI argument.
        """
        if self.args.active_stages:
            self.model.num_stages = self.args.active_stages
        if self.use_cuda:
            import torch.backends.cudnn as cudnn  # noqa: F401
            cudnn.benchmark = True
        loaders = zip(self.cycle_batches(self.train_loader),
                      self.cycle_batches(self.test_loader))
        for i, (train_batch, test_batch) in enumerate(loaders):
            if not i % self.args.step_interval:
                if i:
                    self.update_lr(i)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if self.args.step_limit and i == self.args.step_limit:
                self.save()
                return

            losses = self.model.train_quantities(train_batch,
                                                 checkpoint=self.args.grad_checkpoint,
                                                 epsilon=self.args.epsilon)
            with torch.no_grad():
                test_losses = self.model.train_quantities(test_batch,
                                                          checkpoint=self.args.grad_checkpoint)
            loss = (losses['choice'] +
                    losses['final'] * self.args.final_coeff +
                    losses['all'] * self.args.aux_coeff)
            loss.backward()
            print('step %d: train=%f test=%f entropy=%f used=%d' %
                  (i, losses['final'].item(), test_losses['final'].item(), losses['entropy'],
                   losses['used']))
            if not i % self.args.save_interval:
                self.save()

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

    def save(self):
        """
        Save all the files that this model can produce.
        """
        self.save_checkpoint()
        self.save_reconstructions()
        self.save_samples()

    def update_lr(self, step_idx):
        """
        Set the learning rate of the optimizer for the
        current step number.
        """
        if not self.args.step_limit:
            return
        frac = step_idx / self.args.step_limit
        lr = self.args.lr * (1 - frac) + self.args.lr_final * frac
        for pg in self.optimizer.param_groups:
            if 'lr' in pg:
                pg['lr'] = lr

    def save_checkpoint(self):
        """
        Save a checkpoint of the current model.
        """
        torch.save(self.model.state_dict(), self.args.checkpoint)

    @abstractmethod
    def save_reconstructions(self):
        """
        Save reconstructions to a file.
        """
        pass

    @abstractmethod
    def save_samples(self):
        """
        Save random samples to a file.
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
    def default_segment(self):
        """
        Get the default segment length, or None to
        disable segments.
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Get the shape of the encoded tensors.
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
        Create a new Encoder model.
        """
        pass

    def cycle_batches(self, loader):
        """
        Utility to infinitely cycle through batches from a
        data loader.
        """
        return cycle_batches_simple(loader, self.device)

    def gather_samples(self, loader, num_samples):
        """
        Gather a batch of samples from a data loader.
        """
        return gather_samples_simple(self.cycle_batches(loader), num_samples)


class ImageMixin:
    """
    A mixin to add image-specific functions to a Trainer
    or a Distiller.
    """

    def arg_parser(self):
        res = super().arg_parser()
        res.add_argument('--grid-size', default=5, type=int)
        if self.supports_gaussian:
            res.add_argument('--gaussian', action='store_true')
        return res

    def save_reconstructions(self):
        """
        Save a reconstruction grid to a file.
        """
        data = self.gather_samples(self.test_loader, self.image_grid_size ** 2).to(self.device)
        with torch.no_grad():
            recons = self.model.reconstruct(data)
        recons = self.output_mode(recons)
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
        self.save_grid('samples.png', self.output_mode(img))

    def save_grid(self, path, images):
        """
        Save an arbitrary grid of images, where the images
        are of batch size self.image_grid_size**2, and
        have all but the last dimension in common with
        self.mode_shape.
        """
        img = images.view(self.image_grid_size**2, *self.mode_shape[:-1], -1)
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        img = np.clip(self.denormalize_image(img), 0, 1)
        img = (img * 255).astype('uint8')
        new_shape = [self.image_grid_size, self.shape[1]*self.image_grid_size, -1, self.shape[0]]
        grid = np.concatenate(img.reshape(new_shape), axis=-2).squeeze()
        Image.fromarray(grid).save(path)

    def output_mode(self, outputs):
        if len(outputs.shape) == 5:
            # Only use means of the gaussian.
            return outputs[..., 0].contiguous()
        return outputs

    @property
    def mode_shape(self):
        if len(self.shape) == 4:
            # Only use means of the gaussian.
            return self.shape[:-1]
        return self.shape

    @property
    def image_grid_size(self):
        """
        Determine the number of images to save.
        """
        return self.args.grid_size

    @property
    def supports_gaussian(self):
        """
        Override and return True if gaussian outputs are
        allowed with a flag.
        """
        return False

    @abstractmethod
    def denormalize_image(self, img):
        """
        Turn an image from the model/dataset into a Tensor
        of floats from 0 to 1.
        """
        pass


class TextMixin:
    """
    A mixin to add text-specific functionality to a
    Trainer or Distiller.
    """

    def save_reconstructions(self):
        """
        Save a reconstruction sample to a file.
        """
        data = self.gather_samples(self.test_loader, self.sample_count).to(self.device)
        with torch.no_grad():
            recons = self.model.reconstruct(data)
        texts = [self.tensor_to_string(data[i]) + '\n' + self.tensor_to_string(recons[i])
                 for i in range(data.shape[0])]
        with open('reconstructions.txt', 'w+') as f:
            f.write('\n\n'.join(texts))

    def save_samples(self):
        """
        Save a sample grid to a file.
        """
        latents = torch.randint(high=self.model.options,
                                size=(self.sample_count, self.model.num_stages))
        with torch.no_grad():
            tokens = self.model.decode(latents.to(self.device))
        texts = [self.tensor_to_string(tokens[i]) for i in range(tokens.shape[0])]
        with open('samples.txt', 'w+') as f:
            f.write('\n\n'.join(texts))

    @property
    def sample_count(self):
        """
        Determine the number of samples to save.
        """
        return 5

    def tensor_to_string(self, tensor):
        """
        Convert a tensor of tokens (or logits) into a string.
        """
        if tensor.dtype.is_floating_point:
            tensor = torch.argmax(tensor, dim=-1)
        field = self.train_loader.dataset.fields['text']
        return ''.join(field.vocab.itos[i] for i in tensor.detach().cpu().numpy())

    @property
    def shuffle_data(self):
        """
        If True, gather all the data and shuffle it.
        """
        return True

    def cycle_batches(self, loader):
        """
        Utility to infinitely cycle through batches from a
        data loader.
        """
        if not self.shuffle_data:
            while True:
                for batch in loader:
                    yield batch.text.to(self.device).t()
        else:
            batches = [batch for batch in loader]
            while True:
                random.shuffle(batches)
                for x in batches:
                    yield x.text.to(self.device).t()


class ImageTrainer(ImageMixin, Trainer):
    """
    A Trainer for image datasets.
    """
    pass


class TextTrainer(TextMixin, Trainer):
    """
    A Trainer for text datasets.
    """
    pass


def cycle_batches_simple(loader, device):
    while True:
        for batch, _ in loader:
            yield batch.to(device)


def gather_samples_simple(batches, num_samples):
    results = []
    count = 0
    for inputs in batches:
        results.append(inputs)
        count += inputs.shape[0]
        if count >= num_samples:
            break
    return torch.cat(results, dim=0)[:num_samples]
