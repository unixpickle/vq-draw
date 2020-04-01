import math

import matplotlib.pyplot as plt

# Imported just for the pyplot plugin.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import torch

from vq_draw import BCELoss, Encoder, ModelNetRefiner, SegmentRefiner, Trainer
from vq_draw.modelnet import ModelNetDataset

GRID_SIZE = 64
SAVE_GRID_SIZE = 2


class ModelNetTrainer(Trainer):
    def arg_parser(self):
        res = super().arg_parser()
        res.add_argument('data_dir', type=str)
        return res

    @property
    def default_checkpoint(self):
        return 'modelnet_model.pt'

    @property
    def default_stages(self):
        return 20

    @property
    def default_segment(self):
        return 5

    @property
    def shape(self):
        return (1, GRID_SIZE, GRID_SIZE, GRID_SIZE)

    def create_datasets(self):
        # Taken from pytorch MNIST demo.
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            ModelNetDataset(self.args.data_dir, split='train'),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            ModelNetDataset(self.args.data_dir, split='test'),
            batch_size=self.args.batch, shuffle=True, **kwargs)
        return train_loader, test_loader

    def create_model(self):
        def make_refiner():
            return ModelNetRefiner(self.args.options, self.args.segment)

        num_refiners = int(math.ceil(self.args.stages / self.args.segment))
        refiner = SegmentRefiner(self.args.segment, *[make_refiner() for _ in range(num_refiners)])

        return Encoder(shape=self.shape,
                       options=self.args.options,
                       refiner=refiner,
                       loss_fn=BCELoss())

    def save_reconstructions(self):
        data = self.gather_samples(self.test_loader, SAVE_GRID_SIZE ** 2).to(self.device)
        with torch.no_grad():
            recons = self.model.reconstruct(data)
        img = torch.cat([data, recons], dim=-1)
        self.save_grid(img, 'renderings.png')

    def save_samples(self):
        latents = torch.randint(high=self.model.options,
                                size=(SAVE_GRID_SIZE**2, self.model.num_stages))
        with torch.no_grad():
            tensor = self.model.decode(latents.to(self.device))
        self.save_grid(tensor, 'samples.png')

    def save_grid(self, grid, path):
        print('starting render...')
        grid = grid.view(SAVE_GRID_SIZE, -1, GRID_SIZE, GRID_SIZE)
        grid = (grid > 0).cpu().numpy().astype('bool')
        grid = np.concatenate(grid, axis=1)
        print('grid shape', grid.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(grid)
        plt.savefig(path)
        plt.close()
        print('saving grid...')

    def cycle_batches(self, loader):
        # Override because we do not yield class labels.
        while True:
            for batch in loader:
                yield batch


if __name__ == '__main__':
    ModelNetTrainer().main()
