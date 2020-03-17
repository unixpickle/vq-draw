"""
Implementation of VQ-DRAW.

Original paper: https://arxiv.org/abs/2003.01599.
"""

# flake8: noqa

from .distill import DistillAE, MNISTDistillAE
from .encoder import Encoder
from .losses import MSELoss, SoftmaxLoss, GaussianLoss
from .models import CIFARRefiner, CelebARefiner, MNISTRefiner, SVHNRefiner, TextRefiner
from .refiner import SegmentRefiner, ResidualRefiner
from .train import Trainer, ImageTrainer, TextTrainer, Distiller, ImageDistiller

__all__ = dir()
