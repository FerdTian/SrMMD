import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
os.environ.setdefault("JAX_PLATFORMS", "cpu") 
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import sys
import argparse
import numpy as np
import time
import pickle

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

from kwgflows.gradient_flow import gradient_flow
from kwgflows.divergences.mmd import *
from kwgflows.rkhs.kernels import *
from kwgflows.utils import *
from kwgflows.generate_data import *

def get_config():
    parser = argparse.ArgumentParser(description='SrMMD/DrMMD/HrMMD generative experiments')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--flow', type=str, default=None)
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--spectral', type=str, default='tikhonov')
    parser.add_argument('--lmbda', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--step_size', type=float, default=0.001) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--nystrom', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--adaptive_lmbda', action='store_true')
    parser.add_argument('--save_path', type=str, default='./results_new/')
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=10000)
    parser.add_argument('--source_particle_num', type=int, default=300)
    parser.add_argument('--target_particle_num', type=int, default=300)
    parser.add_argument('--diffusion_noise_scale', type=float, default=1.0)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--logccv', type=float, default=1.0)
    parser.add_argument('--offset', type=float, default=0.0)
    parser.add_argument('--opt', type=str, default='sgd')
    ##### MNIST experiment configs #####
    parser.add_argument('--digits', type=str, default='all') # 'all' or e.g. '0139'
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    if args.flow == 'drmmd_spectral':
        args.save_path += f"{args.dataset}_dataset/{args.kernel}_kernel/{args.flow}_flow/{args.spectral}_spectral/"
        args.save_path += f"__lmbda_{args.lmbda}__alpha_{args.alpha}__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}__offset_{args.offset}"
        args.save_path += f"__source_particle_num_{args.source_particle_num}__inject_noise_scale_{args.inject_noise_scale}"
        args.save_path += f"__opt_{args.opt}__seed_{args.seed}"
    else:
        args.save_path += f"{args.dataset}_dataset/{args.kernel}_kernel/{args.flow}_flow/"
        args.save_path += f"__lmbda_{args.lmbda}__alpha_{args.alpha}__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
        args.save_path += f"__adaptive_lmbda_{str(args.adaptive_lmbda)}"
        args.save_path += f"__source_particle_num_{args.source_particle_num}__inject_noise_scale_{args.inject_noise_scale}__nystrom_{str(args.nystrom)}"
        args.save_path += f"__logccv_{args.logccv}__opt_{args.opt}__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def main(args):
    rng_key = random.PRNGKey(args.seed)
    N = args.source_particle_num
    M = args.target_particle_num
    # assert N == M # Only consider Source and target particle number are the same for now

    if args.dataset == 'MixGaussian':
        mu = jnp.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]]).T
        std = jnp.sqrt(jnp.array([[0.3, 0.3]] * 4)).T
        X, Y, source_dist, target_dist = generate_mog_and_gaussian(args, M, N, mu, std)
    elif args.dataset == 'Gaussian1d':
        X, Y = generate_gaussian1d(args, M, N)
    elif args.dataset == 'Gaussian2d':
        X, Y = generate_gaussian2d(args, Nx=M, Ny=N, std=args.logccv, offset=args.offset)
    elif args.dataset == 'ThreeRing':
        X, Y = generate_three_ring_and_gaussian(args, int(M / 3), N) # Three ring, so divide by 3
    elif args.dataset == 'Student':
        X, Y = generate_student_and_gaussian(args, M, N)
    elif args.dataset == 'swissroll':
        X, Y = generate_swissroll_and_gaussian(args, M, N)
    elif args.dataset == 'checkerboard':
        X, Y = generate_checkerboard_and_gaussian(args, M, N)
    else:
        raise NotImplementedError
    
    if args.kernel == 'Gaussian':
        kernel = gaussian_kernel(args.bandwidth)
    elif args.kernel == 'Laplace':
        kernel = laplace_kernel(args.bandwidth)
    elif args.kernel == 'IMQ':
        kernel = imq_kernel(args.bandwidth)
    elif args.kernel == 'Energy':
        kernel = energy_kernel(1., args.bandwidth, 1e-8)
    elif args.kernel == 'GaussianLinear':
        kernel = gaussian_linear_kernel(args.bandwidth)
    elif args.kernel == 'Stein':
        kernel = stein_kernel(std=args.logccv, sigma=args.bandwidth)
    else:
        raise NotImplementedError
    args.kernel_fn = kernel

    if args.flow == 'mmd':
        divergence = mmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X, Y, args.lmbda)
    elif args.flow == 'drmmd':
        if args.adaptive_lmbda:
            divergence = drmmd_fixed_target_adaptive(args, kernel, None)
            divergence.pre_compute(X, Y, args.lmbda)
        else:
            divergence = drmmd_fixed_target(args, kernel, None)
            divergence.pre_compute(X, Y, args.lmbda)
    elif args.flow == 'srmmd':
        divergence = srmmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X, Y, args.lmbda)
    elif args.flow == 'hrmmd':
        divergence = hrmmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X, Y, args.lmbda)
    else:
        raise NotImplementedError

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # args.save_path += f'{timestamp}/'
    
    ret = gradient_flow(divergence, rng_key, Y, args)
    rate = 100
    # evaluate(args, ret, rate)
    metrics = evaluate(args, ret, rate)
    print(metrics["final_mmd"], metrics["final_w2"])
    if args.dataset == 'Gaussian':
        rate = 10
        # save_animation_1d(args, ret, rate, save_path=args.save_path)
    else:
        # save_animation_2d(args, ret, rate, save_path=args.save_path)
        # save_snapshots_2d(args, ret, save_path=args.save_path, snapshot_iters=[0, 10, 200])
        pass
    pause = True


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')