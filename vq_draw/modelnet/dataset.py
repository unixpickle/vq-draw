import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetDataset(Dataset):
    """
    A Dataset of voxel maps.

    Each sample is a [1 x D x D x D] float Tensor of 1's
    and 0's.

    Args:
        dirname: the voxelized data directory.
        split: either 'train' or 'test'.
    """

    def __init__(self, dirname, split='train'):
        self.paths = []
        for class_dir in os.listdir(dirname):
            split_path = os.path.join(dirname, class_dir, split)
            if not os.path.isdir(split_path):
                raise FileNotFoundError('split directory not present: ' + split_path)
            for file_path in os.listdir(split_path):
                if not file_path.endswith('.npz'):
                    continue
                full_path = os.path.join(split_path, file_path)
                self.paths.append(full_path)
        self.paths = sorted(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.paths[idx])['voxels'])[None].float()
