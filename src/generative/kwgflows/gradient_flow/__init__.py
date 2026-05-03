from typing import Optional, Callable

import optax
import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, random
from jax.tree_util import tree_map
from jax_tqdm import scan_tqdm

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import Divergence, KernelizedDivergence
from kwgflows.typing import Array

# jax.config.update("jax_platform_name", "cpu")


class GradientFlowResult(struct.PyTreeNode):
    divergence: KernelizedDivergence
    Ys: DiscreteProbability

    def get_Yt(self, t):
        return tree_map(lambda x: x[t, :], self.Ys)

    def get_Y_all(self):
        return tree_map(lambda x: x, self.Ys)
    
    def get_velocity_field(self, t, reverse=False):
        Yt = self.get_Yt(t)
        first_variation = self.divergence.get_first_variation(Yt)
        first_variation = jax.lax.cond(
            reverse, lambda _: -first_variation, lambda _: first_variation, None
        )
        return grad(first_variation)


def gradient_flow(
    divergence: KernelizedDivergence,
    rng_key: Array,
    Y: Array,
    args
) -> GradientFlowResult:
    optimizer = optax.sgd(learning_rate=args.step_size)
    opt_state = optimizer.init(Y)
    lambda_list = jnp.zeros(args.step_num)
    
    @scan_tqdm(args.step_num)
    def one_step(dummy, i: Array):
        opt_state, rng_key, Y, lambda_list, lmbda = dummy
        # optimizer = optax.sgd(learning_rate=lmbda * args.step_size)
        optimizer = optax.sgd(learning_rate=args.step_size)

        first_variation = divergence.get_first_variation(Y, lmbda)
        velocity_field = jax.vmap(grad(first_variation))
        updates, new_opt_state = optimizer.update(velocity_field(Y), opt_state)
        Y_next = optax.apply_updates(Y, updates)

        rng_key, _ = random.split(rng_key)
        lambda_list = lambda_list.at[i].set(lmbda)
        dummy_next = (new_opt_state, rng_key, Y_next, lambda_list, lmbda)
        return dummy_next, Y_next

    @scan_tqdm(args.step_num)
    def one_step_adaptive(dummy, i: Array):
        opt_state, rng_key, Y, lambda_list, lmbda = dummy
        optimizer = optax.sgd(learning_rate=jnp.maximum(lmbda * args.step_size, 1e-4))

        first_variation = divergence.get_first_variation(Y, lmbda)
        velocity_field = jax.vmap(grad(first_variation))
        updates, new_opt_state = optimizer.update(velocity_field(Y), opt_state)
        Y_next = optax.apply_updates(Y, updates)

        DrMMD_next = divergence(Y_next, lmbda)
        lmbda = args.lmbda / jnp.sqrt(DrMMD_0 / DrMMD_next)
        # lmbda = lmbda * 0.99

        rng_key, _ = random.split(rng_key)
        lambda_list = lambda_list.at[i].set(lmbda)
        dummy_next = (new_opt_state, rng_key, Y_next, lambda_list, lmbda)
        return dummy_next, Y_next

    if args.adaptive_lmbda:
        DrMMD_0 = divergence(Y, args.lmbda)
        info_dict, trajectory = jax.lax.scan(one_step_adaptive, (opt_state, rng_key, Y, lambda_list, args.lmbda), jnp.arange(args.step_num))
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(info_dict[-2])
        plt.yscale('log')
        plt.savefig(f'{args.save_path}/lambda.png')
        jnp.save(f'{args.save_path}/lambda_array.npy', jnp.array(info_dict[-2]))
        pause = True
    else:
        info_dict, trajectory = jax.lax.scan(one_step, (opt_state, rng_key, Y, lambda_list, args.lmbda), jnp.arange(args.step_num))
    

    return GradientFlowResult(divergence=divergence, Ys=trajectory)
