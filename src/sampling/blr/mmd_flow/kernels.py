import jax.numpy as jnp
from jax import vmap

from .typing import Array, ScoreFunction


class stein_kernel:
    is_stein = True

    def __init__(self, sigma: float, score_fn: ScoreFunction):
        self.sigma = sigma
        self.score_fn = score_fn

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        diff = X[:, None, :] - Y[None, :, :]
        sq_dist = jnp.sum(diff * diff, axis=-1)

        inv_sigma2 = 1.0 / (self.sigma * self.sigma)
        inv_sigma4 = inv_sigma2 * inv_sigma2

        base_kernel = jnp.exp(-0.5 * sq_dist * inv_sigma2)
        score_x = self.score_fn(X)
        score_y = self.score_fn(Y)

        score_dot = jnp.einsum("id,jd->ij", score_x, score_y)
        score_x_grad_y = jnp.einsum("id,ijd->ij", score_x, diff * inv_sigma2)
        grad_x_score_y = jnp.einsum("ijd,jd->ij", -diff * inv_sigma2, score_y)
        trace_hess = (X.shape[-1] * inv_sigma2 - sq_dist * inv_sigma4) * base_kernel

        return score_dot * base_kernel + score_x_grad_y * base_kernel + grad_x_score_y * base_kernel + trace_hess

    def __call__(self, x: Array, y: Array) -> Array:
        dim = x.shape[-1]
        diff = x - y
        sq_dist = jnp.sum(diff * diff)

        inv_sigma2 = 1.0 / (self.sigma * self.sigma)
        inv_sigma4 = inv_sigma2 * inv_sigma2

        base_kernel = jnp.exp(-0.5 * sq_dist * inv_sigma2)
        grad_x = -(diff * inv_sigma2) * base_kernel
        grad_y = +(diff * inv_sigma2) * base_kernel
        trace_hess = (dim * inv_sigma2 - sq_dist * inv_sigma4) * base_kernel

        score_x = self.score_fn(x)
        score_y = self.score_fn(y)

        return (
            jnp.dot(score_x, score_y) * base_kernel
            + jnp.dot(score_x, grad_y)
            + jnp.dot(grad_x, score_y)
            + trace_hess
        )
