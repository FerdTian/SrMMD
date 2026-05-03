import jax.numpy as jnp
import jax
import time
from functools import partial
from .typing import Array, Scalar, Distribution
from typing import Callable

class mmd_fixed_target:
    def __init__(self, args, kernel, distribution):
        self.kernel = kernel
        self.distribution = distribution
        self.args = args
    def _is_stein(self):
        return getattr(self.kernel, "is_stein", False)
    
    def get_witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        if self._is_stein():
            K_zX = jnp.array(0.0)
        else:
            K_zX = self.distribution.mean_embedding(z)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (-K_zX + K_zY.mean(1)).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.get_witness_function, Y=Y)
    
    def __call__(self, Y):
        # K_XX = self.distribution.mean_mean_embedding()
        # K_YY = self.kernel.make_distance_matrix(Y, Y)
        # if self._is_stein():
        #     # Stein kernel: K_XX = 0, K_XY = 0  -> MMD = sqrt(E[k_p(Y,Y')]) = KSD
        #     return jnp.sqrt(jnp.maximum(K_YY, 0.0))
        # K_XY = self.distribution.mean_embedding(Y)
        # return jnp.sqrt(K_XX + K_YY.mean() - 2 * K_XY.mean())
        is_stein = getattr(self.kernel, "is_stein", False)

        K_YY = self.kernel.make_distance_matrix(Y, Y)

        # Stein: K_XX = 0, K_XY = 0
        # K_XX = jnp.array(0.0) if is_stein else self.distribution.mean_mean_embedding()
        # K_XY = jnp.array(0.0) if is_stein else self.distribution.mean_embedding(Y)
        K_XX = self.distribution.mean_mean_embedding()
        K_XY = self.distribution.mean_embedding(Y)
        return jnp.sqrt(K_XX + K_YY.mean() - 2 * K_XY.mean())
    


# -----------------------------
# Autodiff-based helper routines
# -----------------------------
def _scalar_k_from_matrix_fn(kernel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Lift `kernel.make_distance_matrix` (pairwise) into a scalar kernel k(x,y)."""
    # return kernel.make_distance_matrix(x[None, :], y[None, :]).reshape(())
    return kernel(x, y)

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


class srmmd_fixed_target:
    def __init__(self, args, kernel, distribution):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.distribution = distribution
        

    def _is_stein(self):
        return getattr(self.kernel, "is_stein", False)

    def _mean_grad_embedding_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        # Stein kernel: E_X[∇_y k_pi(y,X)] = 0  (regularity assumed)
        if self._is_stein():
            return jnp.zeros((Y.size,), dtype=Y.dtype)

        # def phi(y):
        #     return self.distribution.mean_embedding(y[None, :]).reshape(())
        # grad_phi = jax.jacfwd(phi)
        # G = jax.vmap(grad_phi)(Y)
        # return G.reshape(-1)

    def witness_function(self, z, Y, lmbda):
        """
        Signature matches drmmd_fixed_target.witness_function(z, Y, lmbda).

        X-handling is via distribution mean embedding, analogous to mmd:
            K_zX := E_X[k(z, X)]  via distribution.mean_embedding
        and gradient mean term uses autodiff of mean_embedding.

        NOTE: We keep the rest of the existing SrMMD structure (D_YY, H_YY, cg, etc.)
        and only replace the X-dependent pieces.
        """
        if lmbda is not None:
            self.lmbda = lmbda

        z = z[None, :]
        N = Y.shape[0]

        K_zY = self.kernel.make_distance_matrix(z, Y)  # [1,N]

        # if self._is_stein():
        #     K_zX = jnp.array(0.0)
        # else:
        K_zX = self.distribution.mean_embedding(z).reshape((1,))

        mean_term = K_zY.mean(axis=1) - K_zX  # [1]

        D_YY = _grad1_matrix(self.kernel, Y, Y)
        H_YY = _hess12_matrix(self.kernel, Y, Y)

        H_YY = 0.5 * (H_YY + H_YY.T)  # ensure symmetry

        r_Y = (D_YY @ jnp.ones((N,)) / N)
        r_X = self._mean_grad_embedding_Y(Y)
        r = r_Y - r_X

        reg = self.lmbda * N
        # v, info = jax.scipy.sparse.linalg.cg(
        #     H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-6, maxiter=100
        # )
        v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r)

        dY_z = _grad1_vec(self.kernel, Y, z[0])
        correction = dY_z @ v
        return ((mean_term.squeeze() - correction) / self.lmbda).squeeze()
    
    def get_first_variation(self, Y, lmbda) -> Callable:
        """Signature matches drmmd_fixed_target.get_first_variation(Y, lmbda)."""
        return partial(self.witness_function, Y=Y, lmbda=lmbda)
    
    def __call__(self, Y):
        """
        Signature matches drmmd_fixed_target.__call__(Y).

        X-handling is via distribution mean embeddings like mmd:
            K_XX := E_{X,X'}[k(X,X')]  via mean_mean_embedding()
            K_XY := E_X[k(X, Y)]       via mean_embedding(Y)

        Keep the rest of the logic unchanged, only replace X-dependent blocks.
        """
        N = Y.shape[0]
        K_YY = self.kernel.make_distance_matrix(Y, Y).mean()

        # if self._is_stein():
        #     # Stein: K_XX=0, K_XY=0
        #     base = K_YY
        # else:
        K_XX = self.distribution.mean_mean_embedding()
        K_XY = self.distribution.mean_embedding(Y)
        base = K_YY + K_XX - 2.0 * K_XY.mean()

        D_YY = _grad1_matrix(self.kernel, Y, Y)
        H_YY = _hess12_matrix(self.kernel, Y, Y)

        H_YY = 0.5 * (H_YY + H_YY.T)  # ensure symmetry

        r_Y = (D_YY @ jnp.ones((N,)) / N)
        r_X = self._mean_grad_embedding_Y(Y)
        r = r_Y - r_X

        reg = self.lmbda * N
        # v, info = jax.scipy.sparse.linalg.cg(
        #     H_YY + reg * jnp.eye(H_YY.shape[0]), r, tol=1e-6, maxiter=100
        # )
        v = jax.scipy.linalg.solve(H_YY + reg * jnp.eye(H_YY.shape[0]), r)

        return (base - (r @ v)) / self.lmbda

class hrmmd_fixed_target:
    def __init__(self, args, kernel, distribution):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.alpha = args.alpha
        self.args = args
        self.distribution = distribution
        self.X = None

    def pre_compute(self, X, Y, lmbda):
        del Y
        self.X = X
        if lmbda is not None:
            self.lmbda = lmbda
        return

    def _build_block_system(self, Y):
        """
        mu = Y (varying), pi = X (fixed)

        Assume Stein kernel throughout, so X-side mean terms are zero:
            E_X[k_pi(y, X)] = 0
            E_X[∇_y k_pi(y, X)] = 0

        Alpha-weighted constraint:
            alpha ||∇f||^2_{L2^d(mu)}
          + (1-alpha) ||f||^2_{L2(mu)}
          + lambda ||f||^2_H <= 1

        Hence the empirical block system becomes

            A_alpha =
                [[ (1-alpha) K_YY,               sqrt(alpha(1-alpha)) D_YY^T ],
                 [ sqrt(alpha(1-alpha)) D_YY,   alpha H_YY                 ]]

            b_alpha =
                [ sqrt(1-alpha) * (1/N) K_YY 1_N ;
                  sqrt(alpha)   * (1/N) D_YY 1_N ]
        """
        N = Y.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)
        sqrt_cross = jnp.sqrt(alpha * (1.0 - alpha))

        K_YY = self.kernel.make_distance_matrix(Y, Y)      # [N, N]
        D_YY = _grad1_matrix(self.kernel, Y, Y)            # [N*d, N]
        H_YY = _hess12_matrix(self.kernel, Y, Y)           # [N*d, N*d]
        H_YY = 0.5 * (H_YY + H_YY.T)                       # ensure symmetry

        one_N = jnp.ones((N,), dtype=Y.dtype)

        # Stein target mean terms are zero
        g = (K_YY @ one_N) / N                             # [N]
        r = (D_YY @ one_N) / N                             # [N*d]

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

        return A, b

    def witness_function(self, z, Y, lmbda):
        # assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        if lmbda is not None:
            self.lmbda = lmbda

        z = z[None, :]
        N = Y.shape[0]

        alpha = self.alpha
        sqrt_a = jnp.sqrt(alpha)
        sqrt_1ma = jnp.sqrt(1.0 - alpha)

        K_zY = self.kernel.make_distance_matrix(z, Y)      # [1, N]
        K_zX_mean = jnp.array(0.0, dtype=Y.dtype)

        mean_term = K_zY.mean(axis=1).squeeze() - K_zX_mean

        A, b = self._build_block_system(Y)
        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        v = jax.scipy.linalg.solve(A_reg, b)

        kY_z = K_zY.reshape(-1)                            # [N]
        dY_z = _grad1_vec(self.kernel, Y, z[0])           # [N*d]

        phi_top = sqrt_1ma * kY_z
        phi_bot = sqrt_a * dY_z
        phi_z = jnp.concatenate([phi_top, phi_bot], axis=0)

        correction = phi_z @ v
        return ((mean_term - correction) / self.lmbda).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y):
        # assert self.X is not None, "Call pre_compute(X, Y, lmbda) first."

        N = Y.shape[0]
        K_YY = self.kernel.make_distance_matrix(Y, Y).mean()

        # Stein: E[k(X,X')] = 0, E[k(X,Y)] = 0
        base = K_YY

        A, b = self._build_block_system(Y)
        reg = self.lmbda * N
        A_reg = A + reg * jnp.eye(A.shape[0], dtype=A.dtype)

        v = jax.scipy.linalg.solve(A_reg, b)
        return (base - (b @ v)) / self.lmbda