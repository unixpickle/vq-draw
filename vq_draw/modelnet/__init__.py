"""
This is a small library for reading a voxelized version of
the ModelNet dataset.

The voxelized version of the dataset has the same
structure as the origin, but .off files are now .npz,
where each file contains a [D x D x D] boolean tensor
of voxels.

The original ModelNet40 dataset can be converted using the
export_voxels tool provided in this directory.
This tool was written in Go, and it uses an external (but
still made by unixpickle) library for loading models.

Indices are ordered (x, y, z) with respect to the original
3D models, although the axes don't have universal meaning.
"""

from .dataset import ModelNetDataset

__all__ = ['ModelNetDataset']
