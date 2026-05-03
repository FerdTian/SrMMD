import abc
from typing import Callable

from flax import struct

from kwgflows.base import DiscreteProbability
from kwgflows.rkhs.kernels import Array
from kwgflows.rkhs.rkhs import rkhs_element


class Divergence(struct.PyTreeNode, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, p: DiscreteProbability) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability) -> Callable[[Array], Array]:
        raise NotImplementedError


class KernelizedDivergence(Divergence):
    @abc.abstractmethod
    def get_first_variation(self, p: DiscreteProbability) -> rkhs_element:
        raise NotImplementedError
