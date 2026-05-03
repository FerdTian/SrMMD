import jax
import jax.numpy as jnp
from flax import struct
from jax import vmap, grad
from typing_extensions import Self

from kwgflows.rkhs.kernels import Array, base_kernel


class rkhs_element(struct.PyTreeNode):
    X: Array
    w: Array
    kernel: base_kernel

    def __add__(self, other: Self):
        assert self.kernel is other.kernel
        return type(self)(
            X=jnp.concatenate([self.X, other.X], axis=0),
            w=jnp.concatenate([self.w, other.w], axis=0),
            kernel=self.kernel,
        )

    def __sub__(self, other: Self):
        assert self.kernel is other.kernel
        return type(self)(
            X=jnp.concatenate([self.X, other.X], axis=0),
            w=jnp.concatenate([self.w, -other.w], axis=0),
            kernel=self.kernel,
        )

    def __mul__(self, other: Array):
        # other should be scalar
        assert other.ndim == 0
        return self.replace(w=self.w * other)

    def __neg__(self):
        return self.replace(w=-self.w)

    def __call__(self, x):
        return jnp.dot(
            self.w,
            vmap(type(self.kernel).__call__, (None, None, 0))(self.kernel, x, self.X),
        )

    def rkhs_norm(self, squared=False) -> Array:
        K_xx = self.kernel.make_distance_matrix(self.X, self.X)
        norm_squared = jnp.dot(self.w, jnp.dot(K_xx, self.w))
        return jax.lax.cond(
            squared, lambda _: norm_squared, lambda _: jnp.sqrt(norm_squared), None
        )

    def grad(self, x: Array) -> Array:
        # Small helper function to be able to access velocity fields at inference time by
        # reconstructing them from their respective witness functions
        # Storing velocity fields directly would not have been easy because grad(self.__call__)
        # would yield a non-pytree object. In theory, even more structure could be added
        # since this velocity field is actually RKHS element in H^d.
        # TODO: define types for carthesian products
        return grad(self.__call__)(x)
