import jax
import jax.numpy as jnp
from jax import grad, random
import jaxopt
from .typing import Array, Divergence

try:
    from jax_tqdm import scan_tqdm
except Exception:
    def scan_tqdm(*_args, **_kwargs):
        def decorator(fn):
            return fn
        return decorator


def gradient_flow(
    divergence: Divergence,
    rng_key: Array,
    Y: Array,
    args
):
    def scale(i):
        return jnp.sqrt(1.0 / (i + 1))

    @scan_tqdm(args.step_num)
    def one_step(dummy, i: Array):
        rng_key, Y = dummy

        first_variation = divergence.get_first_variation(Y, lmbda=args.lmbda)
        velocity_field = jax.vmap(grad(first_variation))
        u = random.normal(rng_key, shape=Y.shape)
        beta = args.inject_noise_scale * scale(jnp.squeeze(i))
        Y_next = Y - args.step_size * velocity_field(Y + beta * u)
        rng_key, _ = random.split(rng_key)
        dummy_next = (rng_key, Y_next)
        return dummy_next, Y_next

    info_dict, trajectory = jax.lax.scan(one_step, (rng_key, Y), jnp.arange(args.step_num))
    return info_dict, trajectory
def lbfgs_flow(
    divergence: Divergence,
    rng_key: Array,
    particles: Array,
    args,
):
    del rng_key
    solver = jaxopt.LBFGS(
        fun=divergence,
        value_and_grad=False,
        maxiter=args.step_num,
        tol=args.lbfgs_tol,
        stepsize=args.step_size,
        linesearch=args.lbfgs_linesearch,
        linesearch_init=args.lbfgs_linesearch_init,
        stop_if_linesearch_fails=False,
        maxls=args.lbfgs_max_line_search_iters,
        history_size=args.lbfgs_history_size,
        jit=True,
    )

    current_particles = jnp.asarray(particles)
    state = solver.init_state(current_particles)
    trajectory = []

    for _ in range(args.step_num):
        current_particles, state = solver.update(current_particles, state)
        trajectory.append(current_particles)

    info = {
        "iterations": int(state.iter_num),
        "final_value": float(state.value),
        "final_error": float(state.error),
        "line_search_failures": int(state.failed_linesearch),
        "function_evals": int(state.num_fun_eval),
        "gradient_evals": int(state.num_grad_eval),
        "line_search_iters": int(state.num_linesearch_iter),
    }
    return info, jnp.stack(trajectory, axis=0)
