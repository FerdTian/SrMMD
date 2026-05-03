import abc
from typing import Callable, Type

import jax.numpy as jnp
from flax import struct
from jax import vmap
from typing_extensions import Self

from kwgflows.rkhs.kernels import Array, base_kernel
from kwgflows.rkhs.rkhs import rkhs_element


class DiscreteProbability(struct.PyTreeNode, metaclass=abc.ABCMeta):
    X: Array
    w: Array

    @property
    def num_atoms(self):
        return self.X.shape[0]

    @classmethod
    def from_samples(cls: Type[Self], X: Array):
        assert X.ndim == 2
        num_samples = X.shape[0]
        return cls(X, jnp.ones(num_samples) / num_samples)

    def average_of(self, f: Callable[[Array], Array]) -> Array:
        return jnp.average(vmap(f)(self.X), weights=self.w)

    def push_forward(self, f: Callable[[Array], Array]) -> Self:
        new_Xs = vmap(f)(self.X)
        return self.replace(X=new_Xs)

    def get_mean_embedding(self, kernel: base_kernel) -> rkhs_element:
        return rkhs_element(self.X, self.w, kernel)
