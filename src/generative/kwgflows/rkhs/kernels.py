import abc

import jax.numpy as jnp
from flax import struct
from jax import vmap
from jax import random

from kwgflows.typing import Array


def _rescale(x: Array, scale: Array) -> Array:
    return x / scale


def _l2_norm_squared(x: Array) -> Array:
    return jnp.sum(jnp.square(x))


class base_kernel(struct.PyTreeNode, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )


class gaussian_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-0.5 * _l2_norm_squared(_rescale(x - y, self.sigma)))

class gaussian_linear_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-0.5 * _l2_norm_squared(_rescale(x - y, self.sigma))) + jnp.dot(x, y)

class laplace_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-jnp.sum(jnp.abs(_rescale(x - y, self.sigma))))


class imq_kernel(base_kernel):
    sigma: float
    c: float = 1.0
    beta: float = -0.5

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.power(
            self.c**2 + _l2_norm_squared(_rescale(x - y, self.sigma)), self.beta
        )


class negative_distance_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return -_l2_norm_squared(_rescale(x - y, self.sigma))


class energy_kernel(base_kernel):
    # x0: Array
    beta: float
    sigma: float
    eps: float = 1e-8

    def __call__(self, x: Array, y: Array) -> Array:
        x0 = jnp.zeros_like(x)

        pxx0 = jnp.power(_l2_norm_squared(_rescale(x - x0, self.sigma)) + self.eps, self.beta / 2)
        pyx0 = jnp.power(_l2_norm_squared(_rescale(y - x0, self.sigma)) + self.eps, self.beta / 2)
        pxy = jnp.power(_l2_norm_squared(_rescale(x - y, self.sigma)) + self.eps, self.beta / 2)

        ret = 0.5 * (pxx0 + pyx0 - pxy)
        return ret

class stein_kernel(base_kernel):
    # pi ~ N((0,0), diag(std^2, std^2))
    # base kernel: RBF with bandwidth sigma
    std: float
    sigma: float

    def _score(self, x: Array) -> Array:
        # s(x) = ∇ log π(x) = - x / std^2
        inv_std2 = 1.0 / (self.std * self.std)
        return -x * inv_std2

    def __call__(self, x: Array, y: Array) -> Array:
        # strictly 2D
        if x.shape[-1] != 2 or y.shape[-1] != 2:
            raise ValueError(
                f"stein_kernel expects x,y to be 2D vectors, got {x.shape}, {y.shape}"
            )

        d = x - y
        r2 = jnp.sum(jnp.square(d))

        inv_sigma2 = 1.0 / (self.sigma * self.sigma)
        inv_sigma4 = inv_sigma2 * inv_sigma2

        # RBF kernel
        kxy = jnp.exp(-0.5 * _l2_norm_squared(_rescale(x - y, self.sigma)))

        # ∇_1 k, ∇_2 k
        grad1 = -(d * inv_sigma2) * kxy
        grad2 = +(d * inv_sigma2) * kxy

        # ∇·_1 ∇_2 k  (dim=2)
        div12 = (2.0 * inv_sigma2 - r2 * inv_sigma4) * kxy

        sx = self._score(x)
        sy = self._score(y)

        return (
            jnp.dot(sx, sy) * kxy
            + jnp.dot(sx, grad2)
            + jnp.dot(grad1, sy)
            + div12
        )


class rff_gaussian_kernel(base_kernel):
    """
    Random Fourier Features approximation of the Gaussian RBF kernel:

        k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        k(x,y) ≈ φ(x)^T φ(y)

    where
        w_i ~ N(0, I/sigma^2),  b_i ~ Uniform(0, 2π)
        φ(x) = sqrt(2/m) * cos(W^T x + b)

    Fields:
      - sigma: float
      - W: (d, m) frequencies already scaled by 1/sigma
      - b: (m,) random phases
    """
    sigma: float
    W: Array
    b: Array

    @property
    def num_features(self) -> int:
        return int(self.W.shape[1])

    def features(self, X: Array) -> Array:
        """
        Compute feature map Φ(X):
          X: (n, d) -> (n, m)
        """
        # (n, d) @ (d, m) -> (n, m)
        proj = jnp.matmul(X, self.W) + self.b  # broadcasting b over rows
        scale = jnp.sqrt(2.0 / self.num_features)
        return scale * jnp.cos(proj)

    def __call__(self, x: Array, y: Array) -> Array:
        # x,y: (d,)
        proj_x = jnp.dot(x, self.W) + self.b  # (m,)
        proj_y = jnp.dot(y, self.W) + self.b  # (m,)
        scale = jnp.sqrt(2.0 / self.num_features)
        phi_x = scale * jnp.cos(proj_x)
        phi_y = scale * jnp.cos(proj_y)
        return jnp.dot(phi_x, phi_y)

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        """
        Override to compute Gram matrix efficiently:
          K = Φ(X) Φ(Y)^T
        """
        PhiX = self.features(X)  # (n, m)
        PhiY = self.features(Y)  # (m2, m)
        return jnp.matmul(PhiX, PhiY.T)


def rff_gaussian_kernel_from_key(
    rng_key, *, dim: int, num_features: int, sigma: float
) -> rff_gaussian_kernel:
    """
    Factory for RFF Gaussian kernel.

    Args:
      rng_key: jax.random.PRNGKey
      dim: data dimension d
      num_features: number of random features m
      sigma: RBF bandwidth

    Returns:
      rff_gaussian_kernel instance
    """
    key_w, key_b = random.split(rng_key)
    # W ~ N(0, I/sigma^2) -> sample N(0,1) then divide by sigma
    W = random.normal(key_w, shape=(dim, num_features)) / sigma
    b = random.uniform(key_b, shape=(num_features,), minval=0.0, maxval=2.0 * jnp.pi)
    return rff_gaussian_kernel(sigma=sigma, W=W, b=b)
