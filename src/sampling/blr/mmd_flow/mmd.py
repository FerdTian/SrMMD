from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


def _scalar_k(kernel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return kernel(x, y)


def _grad1_vec(kernel, A: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    k_xy = lambda x, y: _scalar_k(kernel, x, y)
    grad_x = jax.jacfwd(k_xy, argnums=0)
    G = jax.vmap(lambda a_i: grad_x(a_i, z))(A)
    return G.reshape(-1)


def _grad1_matrix(kernel, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    k_xy = lambda x, y: _scalar_k(kernel, x, y)
    grad_x = jax.jacfwd(k_xy, argnums=0)
    G = jax.vmap(lambda a: jax.vmap(lambda b: grad_x(a, b))(B))(A)
    G = jnp.transpose(G, (0, 2, 1))
    A_, d, B_ = G.shape
    return G.reshape(A_ * d, B_)


def _hess12_matrix(kernel, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    k_xy = lambda x, y: _scalar_k(kernel, x, y)

    def hess_block(a, b):
        return jax.jacfwd(jax.jacrev(k_xy, argnums=1), argnums=0)(a, b)

    H = jax.vmap(lambda a: jax.vmap(lambda b: hess_block(a, b))(B))(A)
    H = jnp.transpose(H, (0, 3, 1, 2))
    A_, d, B_, d2 = H.shape
    assert d == d2
    return H.reshape(A_ * d, B_ * d)


class mmd_fixed_target:
    def __init__(self, args, kernel):
        self.kernel = kernel
        self.args = args

    def get_witness_function(self, z, X) -> jnp.ndarray:
        z = z[None, :]
        return self.kernel.make_distance_matrix(z, X).mean(axis=1).squeeze()

    def get_first_variation(self, X, lmbda) -> Callable:
        return partial(self.get_witness_function, X=X)

    def __call__(self, X):
        K_XX = self.kernel.make_distance_matrix(X, X).mean()
        return jnp.sqrt(jnp.maximum(K_XX, 0.0))


class srmmd_fixed_target:
    def __init__(self, args, kernel):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args

    def witness_function(self, z, X, lmbda):
        z = z[None, :]
        N = X.shape[0]

        mean_term = self.kernel.make_distance_matrix(z, X).mean(axis=1)

        D_XX = _grad1_matrix(self.kernel, X, X)
        H_XX = _hess12_matrix(self.kernel, X, X)
        H_XX = 0.5 * (H_XX + H_XX.T)

        r = D_XX @ jnp.ones((N,)) / N
        reg = self.lmbda * N
        v = jax.scipy.linalg.solve(H_XX + reg * jnp.eye(H_XX.shape[0]), r)

        dX_z = _grad1_vec(self.kernel, X, z[0])
        correction = dX_z @ v
        return (mean_term.squeeze() - correction).squeeze() / self.lmbda

    def get_first_variation(self, X, lmbda) -> Callable:
        return partial(self.witness_function, X=X, lmbda=lmbda)

    def __call__(self, X):
        N = X.shape[0]

        base = self.kernel.make_distance_matrix(X, X).mean()

        D_XX = _grad1_matrix(self.kernel, X, X)
        H_XX = _hess12_matrix(self.kernel, X, X)
        H_XX = 0.5 * (H_XX + H_XX.T)

        r = D_XX @ jnp.ones((N,)) / N
        reg = self.lmbda * N
        v = jax.scipy.linalg.solve(H_XX + reg * jnp.eye(H_XX.shape[0]), r)

        return (base - (r @ v)) / self.lmbda
