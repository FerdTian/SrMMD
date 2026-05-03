from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, random
from jax_tqdm import scan_tqdm
import numpy as np
import scipy.optimize

import optax

from .typing import Array, Divergence


def _run_sgd_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    save,
    args,
):
    optimizer = optax.sgd(learning_rate=args.step_size)
    opt_state = optimizer.init(Y)

    threshold = 1e5
    if args.step_num <= threshold:
        step_num = int(args.step_num)
    else:
        step_num = int(threshold)

    def scale(i):
        return jnp.sqrt(1.0 / (i + 1))

    if not save:
        @scan_tqdm(step_num)
        def one_step(dummy, i: Array):
            opt_state, rng_key, Y = dummy
            optimizer = optax.sgd(learning_rate=args.step_size)

            first_variation = divergence.get_first_variation(Y, args.lmbda)
            velocity_field = jax.vmap(grad(first_variation))
            u = jax.random.normal(rng_key, shape=Y.shape)
            beta = args.inject_noise_scale * scale(jnp.squeeze(i))
            updates, new_opt_state = optimizer.update(velocity_field(Y + beta * u), opt_state)
            Y_next = optax.apply_updates(Y, updates)

            rng_key, _ = random.split(rng_key)
            dummy_next = (new_opt_state, rng_key, Y_next)
            return dummy_next, None

        if args.step_num <= threshold:
            info_dict, _ = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(step_num))
            _, _, Y = info_dict
        else:
            for _ in range(int(args.step_num // threshold)):
                info_dict, _ = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(threshold))
                _, _, Y = info_dict
                opt_state = optimizer.init(Y)
                rng_key, _ = random.split(rng_key)
        return info_dict, Y

    @scan_tqdm(step_num)
    def one_step_save_trajectory(dummy, i: Array):
        opt_state, rng_key, Y = dummy
        optimizer = optax.sgd(learning_rate=args.step_size)

        first_variation = divergence.get_first_variation(Y, args.lmbda)
        velocity_field = jax.vmap(grad(first_variation))
        u = jax.random.normal(rng_key, shape=Y.shape)
        beta = args.inject_noise_scale * scale(jnp.squeeze(i))
        updates, new_opt_state = optimizer.update(velocity_field(Y + beta * u), opt_state)
        Y_next = optax.apply_updates(Y, updates)

        rng_key, _ = random.split(rng_key)
        dummy_next = (new_opt_state, rng_key, Y_next)
        return dummy_next, Y_next

    if args.step_num <= threshold:
        info_dict, trajectory = jax.lax.scan(one_step_save_trajectory, (opt_state, rng_key, Y), jnp.arange(step_num))
        return info_dict, trajectory

    save_every = 4
    saved_steps_per_chunk = threshold // save_every
    trajectory_all = np.zeros((args.step_num // save_every, Y.shape[0], Y.shape[1]))

    for iter_idx in range(int(args.step_num // threshold)):
        info_dict, trajectory = jax.lax.scan(one_step_save_trajectory, (opt_state, rng_key, Y), jnp.arange(threshold))
        Y = trajectory[-1, :, :]
        opt_state = optimizer.init(Y)
        rng_key, _ = random.split(rng_key)

        trajectory_subsampled = trajectory[::save_every]
        start_idx = int(iter_idx * saved_steps_per_chunk)
        end_idx = int((iter_idx + 1) * saved_steps_per_chunk)
        trajectory_all[start_idx:end_idx, :, :] = trajectory_subsampled

    return info_dict, trajectory_all


def _run_lbfgs_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    save,
    args,
):
    del rng_key
    y_shape = Y.shape
    y0 = np.asarray(Y).reshape(-1)
    trajectory = []

    @jax.jit
    def value_and_grad_flat(y_flat: Array):
        y_mat = y_flat.reshape(y_shape)
        value = divergence(y_mat)
        grad_val = jax.grad(divergence)(y_mat)
        return value, grad_val.reshape(-1)

    def objective(y_flat_np):
        y_flat = jnp.asarray(y_flat_np).reshape(-1)
        value, grad_val = value_and_grad_flat(y_flat)
        value_np = float(np.asarray(value))
        grad_np = np.asarray(grad_val, dtype=np.float64)
        return value_np, grad_np

    def callback(y_flat_np):
        if save:
            trajectory.append(np.asarray(y_flat_np, dtype=np.float64).reshape(y_shape))

    result = scipy.optimize.minimize(
        fun=objective,
        x0=y0,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={
            'maxiter': int(args.step_num),
            'maxcor': int(getattr(args, 'lbfgs_history_size', 10)),
            'ftol': float(getattr(args, 'lbfgs_ftol', 1e-12)),
            'gtol': float(getattr(args, 'lbfgs_gtol', 1e-8)),
            'maxls': int(getattr(args, 'lbfgs_maxls', 20)),
            'disp': False,
        },
    )

    Y_opt = jnp.asarray(result.x).reshape(y_shape)
    final_value, _ = value_and_grad_flat(jnp.asarray(result.x))
    info_dict = {
        'optimizer': 'lbfgs',
        'success': bool(result.success),
        'status': int(result.status),
        'message': str(result.message),
        'nit': int(result.nit),
        'nfev': int(result.nfev),
        'final_value': float(np.asarray(final_value)),
    }

    if save:
        if len(trajectory) == 0 or not np.allclose(trajectory[-1], np.asarray(Y_opt)):
            trajectory.append(np.asarray(Y_opt))
        return info_dict, jnp.asarray(np.stack(trajectory, axis=0))

    return info_dict, Y_opt


def gradient_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    save,
    args,
):
    optimizer_name = getattr(args, 'optimizer', 'sgd').lower()
    if optimizer_name in ('sgd', 'gd', 'gradient_descent'):
        return _run_sgd_flow(divergence, rng_key, Y, save, args)
    if optimizer_name in ('lbfgs', 'l-bfgs', 'l_bfgs'):
        return _run_lbfgs_flow(divergence, rng_key, Y, save, args)
    raise ValueError(f'Optimizer not recognized: {args.optimizer}')
