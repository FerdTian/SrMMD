import os
import jax.numpy as jnp
from numpyro import distributions as np_distributions
from tensorflow_probability.substrates import jax as tfp
import jax
from torchvision import datasets, transforms, utils as vutils
import numpy as np
import torch

def generate_gaussian1d(args, Nx, Ny):
    rng_key = jax.random.PRNGKey(args.seed)
    bound = 1.0
    tfd = tfp.distributions
    # Define a batch of two scalar TruncatedNormals with modes at 0. and 1.0
    dist = tfd.TruncatedNormal(loc=0.0, scale=1., low=-bound, high=bound)
    X = dist.sample(seed=rng_key, sample_shape=(Nx, 1))
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 1)) * 0.01
    return X, Y

def generate_gaussian2d(args, Nx, Ny, std, offset=0.0):
    rng_key = jax.random.PRNGKey(args.seed)
    X = jax.random.normal(rng_key, (Nx, 2)) * std
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 2)) + offset
    return X, Y

def generate_mog_and_gaussian(args, Nx, Ny, mu, std):
    mixture_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
    target_dist = np_distributions.MixtureSameFamily(
        mixing_distribution=np_distributions.Categorical(probs=mixture_probs),
        component_distribution=np_distributions.Normal(mu, std)
    )

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, _ = jax.random.split(rng_key)
    X = target_dist.sample(rng_key, (Nx,))

    rng_key, _ = jax.random.split(rng_key)
    source_dist = np_distributions.Independent(np_distributions.Normal(jnp.zeros((2,)), 0.1 * jnp.ones((2,))), 1)
    Y = source_dist.sample(rng_key, (Ny,))
    return X, Y, source_dist, target_dist


def generate_three_ring_and_gaussian(args, Nx, Ny):
    rng_key = jax.random.PRNGKey(args.seed)
    r, _delta = 0.3, 0.5
    
    X = jnp.c_[r * jnp.cos(jnp.linspace(0, 2 * jnp.pi, Nx + 1)), r * jnp.sin(jnp.linspace(0, 2 * jnp.pi, Nx + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = jnp.r_[X, X[:Nx, :]-i*jnp.array([0, (2 + _delta) * r])]
    X = jax.random.permutation(rng_key, X)
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 2)) / 100 - jnp.array([0, r])
    return X, Y


def generate_student_and_gaussian(args, Nx, Ny):
    freedom = 2
    rng_key = jax.random.PRNGKey(args.seed)
    
    from scipy.stats import t

    # Sample from two independent t-distributions
    X1 = jnp.array(t.rvs(freedom, size=Nx))
    X2 = jnp.array(t.rvs(freedom, size=Nx))

    # Stack the samples into a 2D array
    samples = jnp.vstack((X1, X2)).T

    # Define a linear transformation matrix A and a translation vector b
    theta = jnp.pi / 4  # 45 degree rotation
    A = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)]])

    X = samples.dot(A)
    # Filter out the samples outside [-threshold, threshold]
    threshold = 10
    mask = (X[:, 0] >= -threshold) & (X[:, 0] <= threshold) & (X[:, 1] >= -threshold) & (X[:, 1] <= threshold)
    X = X[mask]

    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 2)) + jnp.array([2.5, 2.5])
    return X, Y

def generate_swissroll_and_gaussian(args, Nx, Ny, noise=0.02, scale=0.12):
    rng_key = jax.random.PRNGKey(args.seed)

    # target: 2D swiss roll / spiral pattern
    t = jnp.linspace(0.5 * jnp.pi, 4.5 * jnp.pi, Nx)
    r = scale * t
    X = jnp.stack([r * jnp.cos(t), r * jnp.sin(t)], axis=1)

    rng_key, subkey = jax.random.split(rng_key)
    X = X + noise * jax.random.normal(subkey, X.shape)

    rng_key, subkey = jax.random.split(rng_key)
    X = jax.random.permutation(subkey, X, axis=0)

    # source: small Gaussian near the center, similar to other generators
    rng_key, subkey = jax.random.split(rng_key)
    Y = jax.random.normal(subkey, (Ny, 2)) * 0.05

    return X, Y

def generate_checkerboard_and_gaussian(args, Nx, Ny, board_size=4, bound=1.0):
    rng_key = jax.random.PRNGKey(args.seed)

    # 每个小格子的边长
    cell = 2.0 * bound / board_size

    # 先为所有格子编号，挑出“黑格” (i + j) % 2 == 0
    black_cells = []
    for i in range(board_size):
        for j in range(board_size):
            if (i + j) % 2 == 0:
                black_cells.append((i, j))
    black_cells = jnp.array(black_cells)

    n_black = black_cells.shape[0]

    # 给每个样本随机分配一个黑格
    rng_key, subkey = jax.random.split(rng_key)
    cell_ids = jax.random.randint(subkey, (Nx,), 0, n_black)
    chosen_cells = black_cells[cell_ids]   # shape: (Nx, 2)

    # 在对应黑格内均匀采样
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey, (Nx, 2))

    # 左下角坐标
    xy0 = -bound + chosen_cells * cell
    X = xy0 + u * cell

    # source: 小高斯
    rng_key, subkey = jax.random.split(rng_key)
    Y = jax.random.normal(subkey, (Ny, 2)) * 0.05

    return X, Y