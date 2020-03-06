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


class TensorRefinerState(RefinerState):
    """
    A tuple of Tensors.
    """

    def __init__(self, tensors):
        self.tensors = tensors

    def to_tensors(self):
        return tuple(self.tensors)

    def from_tensors(self):
        return lambda x: TensorRefinerState(x)


class TupleRefinerState(RefinerState):
    """
    An aggregate refiner state.
    """

    def __init__(self, states):
        self.states = tuple(states)

    def to_tensors(self):
        return [t for s in self.states for t in s.to_tensors()]

    def from_tensors(self):
        counts = [len(s.to_tensors()) for s in self.states]
        sub_funcs = [s.from_tensors() for s in self.states]

        def f(tensors):
            assert len(tensors) == sum(counts)
            states = []
            for count, func in zip(counts, sub_funcs):
                states.append(func(tensors[:count]))
                tensors = tensors[count:]
            return TupleRefinerState(tuple(states))


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
