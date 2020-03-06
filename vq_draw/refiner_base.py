from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Refiner(nn.Module):
    """
    The base class for all refinement networks.

    A refinement network can be stateful.
    State batches should subclass RefinerState.
    """
    @abstractmethod
    def init_state(self, batch):
        """
        Generate the initial states for a batch, a
        subclass of RefinerState.
        """
        pass

    @abstractmethod
    def forward(self, x, state):
        """
        Apply the network for one timestep.

        Args:
            x: a Tensor of inputs.
            state: the state of the refiner.

        Returns:
            A tuple (options, states):
              options: a [batch x options x ...] Tensor.
              states: another arbitrary state datatype.
        """
        pass


class RefinerState(ABC):
    @abstractmethod
    def to_tensors(self):
        """
        Convert the state to a tuple of differentiable
        Tensors.

        The Tensors must be differentiable to workaround
        shortcomings of gradient checkpointing.
        The gradients needn't be used.
        """
        pass

    @abstractmethod
    def from_tensors(self):
        """
        Return a function that takes Tensors like the ones
        returned by to_tensors() and converts them into a
        RefinerState like self.
        """
        pass


class IntRefinerState(RefinerState):
    """
    An integer-only refiner state.
    """

    def __init__(self, data):
        self.data = data

    def to_tensors(self):
        return (torch.tensor(float(self.data)).requires_grad_(),)

    def from_tensors(self):
        return lambda tensors: IntRefinerState(int(tensors[0].item()))


class ResidualRefiner(Refiner):
    """
    Base class for refiner modules that compute additive
    residuals for the inputs.

    These refiners are assumed to be feed-forward and
    stage-conditioned, so the state is just the stage
    index.
    """
    @abstractmethod
    def residuals(self, x, stage):
        """
        Generate a set of potential deltas to the input.
        """
        pass

    def init_state(self, batch):
        return IntRefinerState(0)

    def forward(self, x, state):
        return (x[:, None] + self.residuals(x, state.data),
                IntRefinerState(state.data + 1))
