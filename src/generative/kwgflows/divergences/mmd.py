from flax import struct
import jax.numpy as jnp
import jax
import time
from functools import partial
from dataclasses import dataclass
from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import KernelizedDivergence
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.typing import Array, Scalar, Distribution
from typing import Callable, Optional


def nystrom_inv(matrix, eps, m):
    rng_key = jax.random.PRNGKey(int(time.time()))
    n = matrix.shape[0]
    m = n
    matrix_mean = jnp.mean(matrix)
    matrix = matrix / matrix_mean  # Scale the matrix to avoid numerical issues

    # Randomly select m columns
    rng_key, _ = jax.random.split(rng_key)
    idx = jax.random.choice(rng_key, n, (m, ), replace=False)

    W = matrix[idx, :][:, idx]
    U, s, V = jnp.linalg.svd(W)

    U_recon = jnp.sqrt(m / n) * matrix[:, idx] @ U @ jnp.diag(1. / s)
    S_recon = s * (n / m)

    Sigma_inv = (1. / eps) * jnp.eye(n)
    approx_inv = Sigma_inv - Sigma_inv @ U_recon @ jnp.linalg.inv(jnp.diag(1. / S_recon) + U_recon.T @ Sigma_inv @ U_recon) @ U_recon.T @ Sigma_inv
    approx_inv = approx_inv / matrix_mean  # Don't forget the scaling!
    return approx_inv

class mmd(struct.PyTreeNode):
    kernel: base_kernel
    
    def get_witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.get_witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar:
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

class mmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g
        # kernel: base_kernel
        # lmbda: float
    
    def pre_compute(self, X, Y, lmbda):
        self.X = X
    
    def get_witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.get_witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y) -> Scalar: # mmd^2
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


class drmmd(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    
    def witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ inv_K_XX @ K_XY).mean(axis=1)
        part3 = K_zX @ inv_K_XX @ K_XX.mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar: # drmmd
        N, M = Y.shape[0], X.shape[0]
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))
        

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ inv_K_XX @ K_XY).mean()
        part3 = (K_XX.T @ inv_K_XX @ K_XY).mean() * 2
        part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)


class drmmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        K_XX = self.kernel.make_distance_matrix(X, X)
        if self.args.nystrom > 0:
            self.K_XX_inv = nystrom_inv(K_XX, self.lmbda, self.args.nystrom)
        else:
            self.K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * self.lmbda * jnp.eye(K_XX.shape[0]))
            # A = K_XX + self.X.shape[0] * self.lmbda * jnp.eye(K_XX.shape[0])
            # L = jnp.linalg.cholesky(A)
            # self.inv_K_XX = jax.scipy.linalg.cho_solve((L, True), jnp.eye(A.shape[0]))
        return
    
    def witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ self.K_XX_inv @ K_XY).mean(axis=1)
        part3 = (K_zX @ self.K_XX_inv @ K_XX).mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y) -> Scalar:
        N, M = Y.shape[0], self.X.shape[0]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ self.K_XX_inv @ K_XY).mean()
        part3 = (K_XX.T @ self.K_XX_inv @ K_XY).mean() * 2
        part4 = -(K_XX.T @ self.K_XX_inv @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)
    

class drmmd_fixed_target_adaptive:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.args = args
        self.g = g

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        self.drmmd = self.__call__(Y, lmbda)
        return
    
    def witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * lmbda * jnp.eye(K_XX.shape[0]))
        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ K_XX_inv @ K_XY).mean(axis=1)
        part3 = (K_zX @ K_XX_inv @ K_XX).mean(axis=1)
        return (part1 + part2 + part3).squeeze() / lmbda * 2 * (1 + lmbda)
    
    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y, lmbda) -> Scalar:
        N, M = Y.shape[0], self.X.shape[0]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ K_XX_inv @ K_XY).mean()
        part3 = (K_XX.T @ K_XX_inv @ K_XY).mean() * 2
        part4 = -(K_XX.T @ K_XX_inv @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / lmbda * (1 + lmbda)


class spectral_drmmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g
        # kernel: base_kernel
        # lmbda: float

    def pre_compute(self, X):
        self.X = X
        M = self.X.shape[0]

        # Centering
        one_M = jnp.ones([M, 1])
        Hs = jnp.eye(M) - one_M @ one_M.T / M
        tilde_Hs = M / (M - 1) * Hs
        from jax.scipy.linalg import sqrtm
        self.tilde_Hs_half = sqrtm(tilde_Hs + 0.000 * jnp.eye(M)).real
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        HKH = (self.tilde_Hs_half.T @ K_XX @ self.tilde_Hs_half) / M
        eig_val, eig_vec = jnp.linalg.eigh(HKH + 1e-10 * jnp.eye(M))
        
        self.G = jnp.zeros([M, M])
        for i in range(M):
            self.G += ( (self.g(eig_val[i]) - self.g(0)) * eig_vec[:, i:i+1] @ eig_vec[:, i:i+1].T ) / eig_val[i]
 
        # Uncentered
        # K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        # eig_val, eig_vec = jnp.linalg.eigh(K_XX / M + 1e-10 * jnp.eye(M))
        # self.G = jnp.zeros([M, M])
        # for i in range(M):
        #     self.G += ( (self.g(eig_val[i]) - self.g(0)) * eig_vec[:, i:i+1] @ eig_vec[:, i:i+1].T ) / eig_val[i]
        return self.G
    
    def witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        # Centered
        part1 = self.g(0) * K_zY.mean(axis=1) + (K_zX @ self.tilde_Hs_half @ self.G @ self.tilde_Hs_half @ K_XY).mean(axis=1) / M
        part2 = - self.g(0) * K_zX.mean(axis=1) - (K_zX @ self.tilde_Hs_half @ self.G @ self.tilde_Hs_half @ K_XX).mean(axis=1) / M

        # Uncentered
        # part1 = self.g(0) * K_zY.mean(axis=1) + (K_zX @ self.G @ K_XY).mean(axis=1) / M
        # part2 = - self.g(0) * K_zX.mean(axis=1) - (K_zX @ self.G @ K_XX).mean(axis=1) / M

        return (part1 + part2).squeeze() * (1 + self.lmbda)
    
    def get_first_variation(self, Y) -> Callable:
        return partial(self.witness_function, Y=Y)

    def __call__(self, Y) -> Scalar:
        return 0
    
class ula(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    X: Array # Target samples
    target_dist: Distribution

    def witness_function(
        self, z
        # In ULA, Y is not needed.
    ) -> Scalar:
        # log_p = self.X.shape[0] * jnp.log(self.std) + jnp.sum(-0.5 * (z - self.mu) ** 2 / self.std ** 2, axis=1)
        log_p = self.target_dist.log_prob(z).sum()
        return -log_p # Energy is negative log density
    
    def get_first_variation(self, Y) -> Callable:
        return self.witness_function




# -----------------------------
# Autodiff-based helper routines
# -----------------------------
def _scalar_k_from_matrix_fn(kernel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Lift `kernel.make_distance_matrix` (pairwise) into a scalar kernel k(x,y)."""
    return kernel.make_distance_matrix(x[None, :], y[None, :]).reshape(())

def _grad1_vec(kernel, A: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Return d_A(z) ∈ ℝ^{(|A|*d)}, entries ∂_{1,ℓ} k(A_i, z) stacked by (i,ℓ)."""
    k_xy = lambda x, y: _scalar_k_from_matrix_fn(kernel, x, y)
    grad_x = jax.jacfwd(k_xy, argnums=0)
    G = jax.vmap(lambda a_i: grad_x(a_i, z))(A)          # [|A|, d]
    return G.reshape(-1)                                  # [|A|*d]

def _grad1_matrix(kernel, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Return D_AB ∈ ℝ^{(|A|*d) × |B|}, entries ∂_{1,ℓ} k(A_i, B_j) flattened along (i,ℓ)."""
    k_xy = lambda x, y: _scalar_k_from_matrix_fn(kernel, x, y)
    grad_x = jax.jacfwd(k_xy, argnums=0)                  # ℝ^d
    G = jax.vmap(lambda a: jax.vmap(lambda b: grad_x(a, b))(B))(A)  # [|A|, |B|, d]
    G = jnp.transpose(G, (0, 2, 1))                       # [|A|, d, |B|]
    A_, d, B_ = G.shape
    return G.reshape(A_ * d, B_)                          # [|A|*d, |B|]

def _hess12_matrix(kernel, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Return H_AB ∈ ℝ^{(|A|*d) × (|B|*d)}, entries ∂_{1,ℓ}∂_{2,m} k(A_i, B_j)
    flattened along (i,ℓ),(j,m).
    """
    k_xy = lambda x, y: _scalar_k_from_matrix_fn(kernel, x, y)
    def hess_block(a, b):
        return jax.jacfwd(jax.jacrev(k_xy, argnums=1), argnums=0)(a, b)  # [d, d]
    H = jax.vmap(lambda a: jax.vmap(lambda b: hess_block(a, b))(B))(A)   # [|A|, |B|, d, d]
    H = jnp.transpose(H, (0, 3, 1, 2))                                   # [|A|, d, |B|, d]
    A_, d, B_, d2 = H.shape
    assert d == d2
    return H.reshape(A_ * d, B_ * d)                                     # [|A|*d, |B|*d]


# ---------------------------------
# SRMMD: match drmmd signatures
# ---------------------------------
class srmmd(struct.PyTreeNode):
    kernel: object
    lmbda: float

    def witness_function(self, z, X, Y):
        """
        Signature matches drmmd.witness_function(z, X, Y).
        Implements:
            f(z) = (1/λ) [ (1/N) Σ k(Y_i, z) - (1/M) Σ k(X_j, z)  -  d_Y(z)^T v ],
        where v = (H_YY + λ N I)^{-1} r,  r = (1/N) D_YY 1_N - (1/M) D_YX 1_M.
        """
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]

        # mean kernel term
        K_zY = self.kernel.make_distance_matrix(z, Y)  # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, X)  # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)  # [1]

        # derivative Gram blocks (μ = Y)
        D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, X)        # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]

        r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)          # [N*d]
        reg = self.lmbda * N
        v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r, assume_a='pos')  # [N*d]

        dY_z = _grad1_vec(self.kernel, Y, z[0])        # [N*d]
        correction = dY_z @ v                           # scalar

        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        """Signature matches drmmd.get_first_variation(X, Y)."""
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y):
        """
        Signature matches drmmd.__call__(X, Y).
        SRMMD^2 = (1/λ) [ ||m_μ - m_π||_H^2 - r^T (H_YY + λ N I)^{-1} r ] with μ=Y, π=X.
        """
        N, M = Y.shape[0], X.shape[0]
        K_YY = self.kernel.make_distance_matrix(Y, Y)  # [N, N]
        K_XX = self.kernel.make_distance_matrix(X, X)  # [M, M]
        K_XY = self.kernel.make_distance_matrix(X, Y)  # [M, N]
        base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()

        D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, X)        # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]
        r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)    # [N*d]
        reg = self.lmbda * N
        v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r, assume_a='pos')
        return (base - (r @ v)) / self.lmbda


# ---------------------------------------------------------
# SRMMD with fixed target: match drmmd_fixed_target signatures
# Here we choose μ = X (fixed), π = Y (varying). That way the
# precomputation depends on X only, mirroring drmmd_fixed_target.
# ---------------------------------------------------------
class srmmd_fixed_target:
    def __init__(self, args, kernel, g):
        """
        Signature matches drmmd_fixed_target.__init__(args, kernel, g).
        `g` is accepted for interface parity (unused here).
        """
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g

        self.X = None
        self.H_XX_inv = None

    def pre_compute(self, X, Y, lmbda):
        """
        Signature matches drmmd_fixed_target.pre_compute(X, Y, lmbda).
        We fix μ = X. Precompute (H_XX + λ M I)^{-1}. Y,lmbda accepted for parity.
        """
        del Y  # unused (parity only)
        if lmbda is not None:
            self.lmbda = lmbda

        self.X = X
        M = self.X.shape[0]
        H_XX = _hess12_matrix(self.kernel, self.X, self.X)   # [M*d, M*d]
        reg = self.lmbda * M
        A = H_XX + reg * jnp.eye(H_XX.shape[0])

        # # Optional Nyström if the project provides it
        # if hasattr(self.args, "nystrom") and getattr(self.args, "nystrom") and self.args.nystrom > 0:
        #     try:
        #         from mmd import nystrom_inv  # must be available in the project
        #         self.H_XX_inv = nystrom_inv(A, 0.0, self.args.nystrom)
        #     except Exception:
        #         self.H_XX_inv = jnp.linalg.inv(A)
        # else:
        #     self.H_XX_inv = jnp.linalg.inv(A)
        return

    def witness_function(self, z, Y, lmbda):
        """
        Signature matches drmmd_fixed_target.witness_function(z, Y, lmbda).
        Uses precomputed μ = X (fixed).
        f(z) = (1/λ) [ (1/M) Σ k(X_j, z) - (1/N) Σ k(Y_i, z)  -  d_X(z)^T v ],
        where v = (H_XX + λ M I)^{-1} r,  r = (1/M) D_XX 1_M - (1/N) D_XY 1_N.
        """
        # assert self.X is not None and self.H_XX_inv is not None, "Call pre_compute(X, Y, lmbda) first."
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."
        # lmbda is accepted for parity; we use the precomputed inverse built with self.lmbda
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]

        # mean kernel term
        K_zY = self.kernel.make_distance_matrix(z, Y)  # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, self.X)  # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)  # [1]

        # derivative Gram blocks (μ = Y)
        D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, self.X)        # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]

        r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)          # [N*d]
        reg = self.lmbda * N
        # v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r)  # [N*d]
        v, info = jax.scipy.sparse.linalg.cg(H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-8, maxiter=10000)

        dY_z = _grad1_vec(self.kernel, Y, z[0])        # [N*d]
        correction = dY_z @ v                           # scalar

        return ((mean_term.squeeze() + correction) / self.lmbda).squeeze()


    def get_first_variation(self, Y, lmbda) -> Callable:
        """Signature matches drmmd_fixed_target.get_first_variation(Y, lmbda)."""
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y):
        """
        Signature matches drmmd_fixed_target.__call__(Y).
        SRMMD^2 with μ = X fixed:
            (1/λ)[ K_YY.mean()+K_XX.mean()-2K_XY.mean() - r^T (H_XX+λ M I)^{-1} r ].
        """
        # assert self.X is not None and self.H_XX_inv is not None, "Call pre_compute(X, Y, lmbda) first."
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."
        N, M = Y.shape[0], self.X.shape[0]
        K_YY = self.kernel.make_distance_matrix(Y, Y)  # [N, N]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)  # [M, M]
        K_XY = self.kernel.make_distance_matrix(self.X, Y)  # [M, N]
        base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()

        D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, self.X)        # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]
        r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)    # [N*d]
        reg = self.lmbda * N
        # v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r)
        v, info = jax.scipy.sparse.linalg.cg(H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-8, maxiter=10000)
        return (base - (r @ v)) / self.lmbda



# # ---------------------------------------------------------
# # SRMMD with fixed target: match drmmd_fixed_target signatures
# # Here we choose μ = X (fixed), π = Y (varying). That way the
# # precomputation depends on X only, mirroring drmmd_fixed_target.
# # ---------------------------------------------------------
# class srmmd_fixed_target:
#     def __init__(self, args, kernel, g):
#         """
#         Signature matches drmmd_fixed_target.__init__(args, kernel, g).
#         `g` is accepted for interface parity (unused here).
#         """
#         self.kernel = kernel
#         self.lmbda = args.lmbda
#         self.args = args
#         self.g = g

#         self.X = None
#         self.H_XX_inv = None

#     def pre_compute(self, X, Y, lmbda):
#         """
#         Signature matches drmmd_fixed_target.pre_compute(X, Y, lmbda).
#         We fix μ = X. Precompute (H_XX + λ M I)^{-1}. Y,lmbda accepted for parity.
#         """
#         del Y  # unused (parity only)
#         if lmbda is not None:
#             self.lmbda = lmbda

#         self.X = X
#         M = self.X.shape[0]
#         H_XX = _hess12_matrix(self.kernel, self.X, self.X)   # [M*d, M*d]
#         reg = self.lmbda * M
#         A = H_XX + reg * jnp.eye(H_XX.shape[0])

#         # # Optional Nyström if the project provides it
#         # if hasattr(self.args, "nystrom") and getattr(self.args, "nystrom") and self.args.nystrom > 0:
#         #     try:
#         #         from mmd import nystrom_inv  # must be available in the project
#         #         self.H_XX_inv = nystrom_inv(A, 0.0, self.args.nystrom)
#         #     except Exception:
#         #         self.H_XX_inv = jnp.linalg.inv(A)
#         # else:
#         #     self.H_XX_inv = jnp.linalg.inv(A)
#         return

#     def witness_function(self, z, Y, lmbda):
#         """
#         Signature matches drmmd_fixed_target.witness_function(z, Y, lmbda).
#         Uses precomputed μ = X (fixed).
#         f(z) = (1/λ) [ (1/M) Σ k(X_j, z) - (1/N) Σ k(Y_i, z)  -  d_X(z)^T v ],
#         where v = (H_XX + λ M I)^{-1} r,  r = (1/M) D_XX 1_M - (1/N) D_XY 1_N.
#         """
#         # assert self.X is not None and self.H_XX_inv is not None, "Call pre_compute(X, Y, lmbda) first."
#         assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."
#         # lmbda is accepted for parity; we use the precomputed inverse built with self.lmbda
#         z = z[None, :]
#         N, M = Y.shape[0], self.X.shape[0]

#         # mean kernel term
#         K_zY = self.kernel.make_distance_matrix(z, Y)  # [1, N]
#         # K_zX = self.kernel.make_distance_matrix(z, self.X)  # [1, M]
#         # mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)  # [1]
#         mean_term = K_zY.mean(axis=1)  # [1]
#         # derivative Gram blocks (μ = Y)
#         D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
#         # D_YX = _grad1_matrix(self.kernel, Y, self.X)        # [N*d, M]
#         H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]

#         # r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)          # [N*d]
#         r = (D_YY @ jnp.ones((N,)) / N) 
#         reg = self.lmbda * N
#         # v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r, assume_a='pos')  # [N*d]
#         v, info = jax.scipy.sparse.linalg.cg(H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-6, maxiter=100)

#         dY_z = _grad1_vec(self.kernel, Y, z[0])        # [N*d]
#         correction = dY_z @ v                           # scalar

#         return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()


#     def get_first_variation(self, Y, lmbda) -> Callable:
#         """Signature matches drmmd_fixed_target.get_first_variation(Y, lmbda)."""
#         return partial(self.witness_function, Y=Y, lmbda=lmbda)

#     def __call__(self, Y):
#         """
#         Signature matches drmmd_fixed_target.__call__(Y).
#         SRMMD^2 with μ = X fixed:
#             (1/λ)[ K_YY.mean()+K_XX.mean()-2K_XY.mean() - r^T (H_XX+λ M I)^{-1} r ].
#         """
#         # assert self.X is not None and self.H_XX_inv is not None, "Call pre_compute(X, Y, lmbda) first."
#         assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."
#         N, M = Y.shape[0], self.X.shape[0]
#         K_YY = self.kernel.make_distance_matrix(Y, Y)  # [N, N]
#         # K_XX = self.kernel.make_distance_matrix(self.X, self.X)  # [M, M]
#         # K_XY = self.kernel.make_distance_matrix(self.X, Y)  # [M, N]
#         # base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()
#         base = K_YY.mean()

#         D_YY = _grad1_matrix(self.kernel, Y, Y)        # [N*d, N]
#         # D_YX = _grad1_matrix(self.kernel, Y, self.X)        # [N*d, M]
#         H_YY = _hess12_matrix(self.kernel, Y, Y)       # [N*d, N*d]
#         # r = (D_YY @ jnp.ones((N,)) / N) - (D_YX @ jnp.ones((M,)) / M)    # [N*d]
#         r = (D_YY @ jnp.ones((N,)) / N)
#         reg = self.lmbda * N
#         # v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r, assume_a='pos')
#         v, info = jax.scipy.sparse.linalg.cg(H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-6, maxiter=100)
#         return (base - (r @ v)) / self.lmbda

# ---------------------------------
# WHSRMMD / alpha-HRMMD:
# norm constraint:
#   alpha ||∇f||^2_{L2^d(mu)}
# + (1-alpha) ||f||^2_{L2(mu)}
# + lambda ||f||^2_H <= 1
#
# witness:
#   f_{mu,pi}^{(alpha)}
#   = (alpha S_mu + (1-alpha) C_mu + lambda I)^(-1) (m_mu - m_pi)
# ---------------------------------
class hrmmd(struct.PyTreeNode):
    kernel: object
    lmbda: float
    alpha: float

    def _build_block_system(self, Y, X):
        """
        mu = Y (current particles), pi = X (target samples)

        Weighted block system for alpha-version:
            A_alpha =
                [[ (1-alpha) K_YY,               sqrt(alpha(1-alpha)) D_YY^T ],
                 [ sqrt(alpha(1-alpha)) D_YY,   alpha H_YY                 ]]

            b_alpha =
                [ sqrt(1-alpha) * ((1/N) K_YY 1_N - (1/M) K_YX 1_M) ;
                  sqrt(alpha)   * ((1/N) D_YY 1_N - (1/M) D_YX 1_M) ]
        """
        N, M = Y.shape[0], X.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)
        sqrt_cross = jnp.sqrt(alpha * (1.0 - alpha))

        K_YY = self.kernel.make_distance_matrix(Y, Y)      # [N, N]
        K_YX = self.kernel.make_distance_matrix(Y, X)      # [N, M]
        D_YY = _grad1_matrix(self.kernel, Y, Y)            # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, X)            # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)           # [N*d, N*d]

        one_N = jnp.ones((N,), dtype=K_YY.dtype)
        one_M = jnp.ones((M,), dtype=K_YY.dtype)

        g = (K_YY @ one_N) / N - (K_YX @ one_M) / M        # [N]
        r = (D_YY @ one_N) / N - (D_YX @ one_M) / M        # [N*d]

        b_top = sqrt_1ma * g                               # [N]
        b_bot = sqrt_a * r                                 # [N*d]
        b = jnp.concatenate([b_top, b_bot], axis=0)        # [N + N*d]

        A_top = jnp.concatenate(
            [(1.0 - alpha) * K_YY, sqrt_cross * D_YY.T], axis=1
        )                                                  # [N, N + N*d]
        A_bot = jnp.concatenate(
            [sqrt_cross * D_YY, alpha * H_YY], axis=1
        )                                                  # [N*d, N + N*d]
        A = jnp.concatenate([A_top, A_bot], axis=0)        # [N + N*d, N + N*d]

        return A, b, K_YY, K_YX, D_YY, D_YX, H_YY

    def witness_function(self, z, X, Y):
        """
        Signature matches srmmd/drmmd:
            witness_function(z, X, Y)

        Alpha-weighted witness:
            f(z) = (1/lambda) [ mean_term(z) - phi_alpha(z)^T v ]
        where
            v = (A_alpha + N lambda I)^(-1) b_alpha

            phi_alpha(z) =
                [ sqrt(1-alpha) * k_Y(z) ;
                  sqrt(alpha)   * d_Y(z) ]
        """
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)

        # Mean kernel term: same sign convention as srmmd
        K_zY = self.kernel.make_distance_matrix(z, Y)      # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, X)      # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)  # [1]

        # Build weighted block system
        A, b, _, _, _, _, _ = self._build_block_system(Y, X)

        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        # Solve
        v = jax.scipy.linalg.solve(A_reg, b, assume_a='pos')

        # Weighted feature vector at z
        kY_z = K_zY.reshape(-1)                            # [N]
        dY_z = _grad1_vec(self.kernel, Y, z[0])           # [N*d]

        phi_top = sqrt_1ma * kY_z
        phi_bot = sqrt_a * dY_z
        phi_z = jnp.concatenate([phi_top, phi_bot], axis=0)

        correction = phi_z @ v
        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y):
        """
        Alpha-weighted HRMMD^2 / WHSRMMD^2:
            (1/lambda) [ MMD^2(X,Y) - b_alpha^T (A_alpha + N lambda I)^(-1) b_alpha ]
        with mu = Y, pi = X.
        """
        N, M = Y.shape[0], X.shape[0]

        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()

        A, b, _, _, _, _, _ = self._build_block_system(Y, X)
        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        v = jax.scipy.linalg.solve(A_reg, b, assume_a='pos')
        return (base - (b @ v)) / self.lmbda

# ---------------------------------------------------------
# Alpha-weighted HRMMD / WHSRMMD with fixed target
# Same calling convention as srmmd_fixed_target:
#   pre_compute(X, Y, lmbda)
#   witness_function(z, Y, lmbda)
#   __call__(Y)
# and use mu = Y, pi = X, i.e. X is fixed target.
# ---------------------------------------------------------
class hrmmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.args = args
        self.g = g

        self.X = None

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        if lmbda is not None:
            self.lmbda = lmbda
        return

    def _build_block_system(self, Y):
        """
        mu = Y (varying), pi = X (fixed)

        Weighted block system:
            A_alpha =
                [[ (1-alpha) K_YY,               sqrt(alpha(1-alpha)) D_YY^T ],
                 [ sqrt(alpha(1-alpha)) D_YY,   alpha H_YY                 ]]

            b_alpha =
                [ sqrt(1-alpha) * ((1/N) K_YY 1_N - (1/M) K_YX 1_M) ;
                  sqrt(alpha)   * ((1/N) D_YY 1_N - (1/M) D_YX 1_M) ]
        """
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        N, M = Y.shape[0], self.X.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)
        sqrt_cross = jnp.sqrt(alpha * (1.0 - alpha))

        K_YY = self.kernel.make_distance_matrix(Y, Y)          # [N, N]
        K_YX = self.kernel.make_distance_matrix(Y, self.X)     # [N, M]
        D_YY = _grad1_matrix(self.kernel, Y, Y)                # [N*d, N]
        D_YX = _grad1_matrix(self.kernel, Y, self.X)           # [N*d, M]
        H_YY = _hess12_matrix(self.kernel, Y, Y)               # [N*d, N*d]

        one_N = jnp.ones((N,), dtype=K_YY.dtype)
        one_M = jnp.ones((M,), dtype=K_YY.dtype)

        g = (K_YY @ one_N) / N - (K_YX @ one_M) / M            # [N]
        r = (D_YY @ one_N) / N - (D_YX @ one_M) / M            # [N*d]

        b_top = sqrt_1ma * g
        b_bot = sqrt_a * r
        b = jnp.concatenate([b_top, b_bot], axis=0)

        A_top = jnp.concatenate(
            [(1.0 - alpha) * K_YY, sqrt_cross * D_YY.T], axis=1
        )
        A_bot = jnp.concatenate(
            [sqrt_cross * D_YY, alpha * H_YY], axis=1
        )
        A = jnp.concatenate([A_top, A_bot], axis=0)

        return A, b, K_YY, K_YX, D_YY, D_YX, H_YY

    def witness_function(self, z, Y, lmbda):
        """
        Signature matches srmmd_fixed_target.witness_function(z, Y, lmbda)
        """
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)

        K_zY = self.kernel.make_distance_matrix(z, Y)          # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, self.X)     # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)

        A, b, _, _, _, _, _ = self._build_block_system(Y)
        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        # dense solve
        # v = jax.scipy.linalg.solve(A_reg, b, assume_a='pos')

        # or CG, same style as srmmd_fixed_target
        v, info = jax.scipy.sparse.linalg.cg(A_reg, b, tol=1e-8, maxiter=10000)

        kY_z = K_zY.reshape(-1)                                # [N]
        dY_z = _grad1_vec(self.kernel, Y, z[0])               # [N*d]

        phi_top = sqrt_1ma * kY_z
        phi_bot = sqrt_a * dY_z
        phi_z = jnp.concatenate([phi_top, phi_bot], axis=0)

        correction = phi_z @ v
        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y):
        """
        Signature matches srmmd_fixed_target.__call__(Y)
        """
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        N, M = Y.shape[0], self.X.shape[0]
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()

        A, b, _, _, _, _, _ = self._build_block_system(Y)
        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        # dense solve
        # v = jax.scipy.linalg.solve(A_reg, b, assume_a='pos')

        v, info = jax.scipy.sparse.linalg.cg(A_reg, b, tol=1e-8, maxiter=10000)
        return (base - (b @ v)) / self.lmbda

# ---------------------------------
# L2-MMD: replace gradient penalty by L2(mu) penalty
# norm constraint:
#   ||f||^2_{L2(mu)} + λ ||f||^2_H <= 1
#
# With mu = Y, pi = X, the witness is
#   f(z) = (1/λ) [ mean_Y k(y,z) - mean_X k(x,z) - k_Y(z)^T v ],
# where
#   v = (K_YY + N λ I)^(-1) r,
#   r = (1/N) K_YY 1_N - (1/M) K_YX 1_M.
#
# and
#   L2MMD^2 = (1/λ) [ MMD^2(X,Y) - r^T (K_YY + N λ I)^(-1) r ].
# ---------------------------------
class l2mmd(struct.PyTreeNode):
    kernel: object
    lmbda: float

    def witness_function(self, z, X, Y):
        """
        Signature matches srmmd/hrmmd:
            witness_function(z, X, Y)

        Convention:
            mu = Y (current particles)
            pi = X (target samples)
        """
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]

        # mean embedding term
        K_zY = self.kernel.make_distance_matrix(z, Y)      # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, X)      # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)  # [1]

        # Gram blocks
        K_YY = self.kernel.make_distance_matrix(Y, Y)      # [N, N]
        K_YX = self.kernel.make_distance_matrix(Y, X)      # [N, M]

        one_N = jnp.ones((N,))
        one_M = jnp.ones((M,))

        r = (K_YY @ one_N) / N - (K_YX @ one_M) / M        # [N]
        reg = self.lmbda * N

        # solve (K_YY + N λ I) v = r
        v = jax.scipy.linalg.solve(
            K_YY + reg * jnp.eye(N),
            r,
            assume_a='pos'
        )                                                  # [N]

        # correction term: k_Y(z)^T v
        correction = (K_zY.reshape(-1) @ v)                # scalar

        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y):
        """
        L2MMD^2 with mu = Y, pi = X:
            (1/λ) [ MMD^2(X,Y) - r^T (K_YY + N λ I)^(-1) r ]
        """
        N, M = Y.shape[0], X.shape[0]

        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)

        base = K_YY.mean() + K_XX.mean() - 2.0 * K_XY.mean()

        K_YX = K_XY.T
        one_N = jnp.ones((N,))
        one_M = jnp.ones((M,))

        r = (K_YY @ one_N) / N - (K_YX @ one_M) / M
        reg = self.lmbda * N

        v = jax.scipy.linalg.solve(
            K_YY + reg * jnp.eye(N),
            r,
            assume_a='pos'
        )
        return (base - (r @ v)) / self.lmbda

# ---------------------------------------------------------
# L2-MMD with fixed target: match srmmd_fixed_target signatures
# Keep the same convention as srmmd_fixed_target:
#   pre_compute(X, Y, lmbda)
#   witness_function(z, Y, lmbda)
#   __call__(Y)
#
# Here:
#   mu = Y (varying particles)
#   pi = X (fixed target)
# ---------------------------------------------------------
class l2mmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g

        self.X = None
        self.K_XX = None
        self.one_M = None

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        if lmbda is not None:
            self.lmbda = lmbda

        self.K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        self.one_M = jnp.ones((self.X.shape[0],))
        return

    def witness_function(self, z, Y, lmbda):
        """
        Signature matches srmmd_fixed_target.witness_function(z, Y, lmbda)
        """
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]

        K_zY = self.kernel.make_distance_matrix(z, Y)          # [1, N]
        K_zX = self.kernel.make_distance_matrix(z, self.X)     # [1, M]
        mean_term = K_zY.mean(axis=1) - K_zX.mean(axis=1)      # [1]

        K_YY = self.kernel.make_distance_matrix(Y, Y)          # [N, N]
        K_YX = self.kernel.make_distance_matrix(Y, self.X)     # [N, M]

        one_N = jnp.ones((N,))
        r = (K_YY @ one_N) / N - (K_YX @ self.one_M) / M       # [N]

        reg = self.lmbda * N

        # dense solve
        # v = jax.scipy.linalg.solve(K_YY + reg * jnp.eye(N), r, assume_a='pos')

        # or CG, consistent with srmmd_fixed_target style
        v, info = jax.scipy.sparse.linalg.cg(
            K_YY + reg * jnp.eye(N),
            r,
            tol=1e-8,
            maxiter=10000
        )

        correction = K_zY.reshape(-1) @ v
        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y):
        """
        Signature matches srmmd_fixed_target.__call__(Y)
        """
        assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        N, M = Y.shape[0], self.X.shape[0]

        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YX = K_XY.T

        base = K_YY.mean() + self.K_XX.mean() - 2.0 * K_XY.mean()

        one_N = jnp.ones((N,))
        r = (K_YY @ one_N) / N - (K_YX @ self.one_M) / M

        reg = self.lmbda * N

        # dense solve
        # v = jax.scipy.linalg.solve(K_YY + reg * jnp.eye(N), r, assume_a='pos')

        v, info = jax.scipy.sparse.linalg.cg(
            K_YY + reg * jnp.eye(N),
            r,
            tol=1e-8,
            maxiter=10000
        )

        return (base - (r @ v)) / self.lmbda