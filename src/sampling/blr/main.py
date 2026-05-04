import argparse
import json
import os
import pickle
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from datasets import BLR_DATASETS, load_blr_dataset
from mmd_flow.blr import BayesianLogisticRegressionTarget
from mmd_flow.gradient_flow import gradient_flow, lbfgs_flow
from mmd_flow.kernels import stein_kernel
from mmd_flow.mmd import mmd_fixed_target, srmmd_fixed_target
from mmd_flow.utils import save_metric_curves

jax.config.update("jax_enable_x64", True)


def build_arg_parser(
    method_choices=("mmd", "srmmd"),
    default_method="srmmd",
    description="SrMMD flow for Bayesian logistic regression posterior sampling",
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=BLR_DATASETS, default="breast_cancer")
    parser.add_argument("--method", choices=method_choices, default=default_method)
    parser.add_argument("--optimizer", choices=("euler", "lbfgs"), default="euler")
    parser.add_argument("--kernel", default="Stein")
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--step_num", type=int, default=2000)
    parser.add_argument("--particle_num", type=int, default=50)
    parser.add_argument("--inject_noise_scale", type=float, default=0.0)
    parser.add_argument("--metric_eval_stride", type=int, default=10)

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--alpha_shape", type=float, default=1.0)
    parser.add_argument("--alpha_rate", type=float, default=0.01)
    parser.add_argument("--covtype_variant", choices=("binary12", "one_vs_rest"), default="binary12")
    parser.add_argument("--covtype_positive_label", type=int, default=2)
    parser.add_argument("--max_train_size", type=int, default=20000)
    parser.add_argument("--max_test_size", type=int, default=10000)
    parser.add_argument("--dataset_cache_dir", type=str, default="./data_cache")

    parser.add_argument("--weight_init_scale", type=float, default=0.05)
    parser.add_argument("--log_alpha_init_scale", type=float, default=0.1)
    parser.add_argument("--lbfgs_history_size", type=int, default=10)
    parser.add_argument("--lbfgs_tol", type=float, default=1e-3)
    parser.add_argument("--lbfgs_linesearch", choices=("zoom", "backtracking"), default="zoom")
    parser.add_argument("--lbfgs_linesearch_init", choices=("increase", "max", "current"), default="increase")
    parser.add_argument("--lbfgs_max_line_search_iters", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./results/")
    return parser


def get_config():
    return build_arg_parser().parse_args()


def _format_float(value):
    return np.format_float_positional(value, trim="-")


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())

    args.save_path = os.path.join(
        args.save_path,
        f"{args.method}_blr_flow",
        args.dataset,
        (
            f"step_size_{_format_float(args.step_size)}"
            f"__bandwidth_{_format_float(args.bandwidth)}"
            f"__lambda_{_format_float(args.lmbda)}"
            f"__step_num_{args.step_num}"
            f"__particle_num_{args.particle_num}"
            f"__alpha_shape_{_format_float(args.alpha_shape)}"
            f"__alpha_rate_{_format_float(args.alpha_rate)}"
            f"__seed_{args.seed}"
            f"__optimizer_{args.optimizer}"
        ),
    )
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/configs.pkl", "wb") as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def build_divergence(args, kernel):
    divergence_cls = srmmd_fixed_target if args.method == "srmmd" else mmd_fixed_target
    return divergence_cls(args, kernel)


def run_particle_optimizer(divergence, rng_key, initial_particles, args):
    if args.optimizer == "lbfgs":
        return lbfgs_flow(divergence, rng_key, initial_particles, args)
    _, trajectory = gradient_flow(divergence, rng_key, initial_particles, args)
    return {}, trajectory


def evaluate_posterior_particles(particles, X, y):
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    particle_array = np.asarray(particles)
    weight_particles = particle_array[:, :-1]
    logits = np.clip(X_aug @ weight_particles.T, -60.0, 60.0)
    sample_probs = 1.0 / (1.0 + np.exp(-logits))
    predictive_probs = np.clip(sample_probs.mean(axis=1), 1e-12, 1.0 - 1e-12)

    mean_log_likelihood = np.mean(
        y * np.log(predictive_probs) + (1.0 - y) * np.log(1.0 - predictive_probs)
    )
    averaged_accuracy = np.mean((predictive_probs >= 0.5) == y)
    mean_alpha = float(np.exp(np.clip(particle_array[:, -1], -20.0, 20.0)).mean())
    finite_fraction = float(np.isfinite(particle_array).mean())
    return {
        "mean_log_likelihood": float(mean_log_likelihood),
        "averaged_accuracy": float(averaged_accuracy),
        "mean_alpha": mean_alpha,
        "finite_fraction": finite_fraction,
    }


def run_blr_flow(args):
    print(f"[1/6] Loading dataset '{args.dataset}'...")
    try:
        X_train, X_test, y_train, y_test, metadata = load_blr_dataset(args)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{args.dataset}'. "
            "The BLR datasets are downloaded into the dataset cache directory on first use, "
            "so network access is required unless they are already cached locally."
        ) from exc
    print(
        "[1/6] Dataset loaded. "
        f"train={metadata['train_size']}, test={metadata['test_size']}, feature_dim={metadata['feature_dim']}"
    )

    print("[2/6] Building BLR posterior target...")
    target = BayesianLogisticRegressionTarget(
        X_train=X_train,
        y_train=y_train,
        alpha_shape=args.alpha_shape,
        alpha_rate=args.alpha_rate,
    )
    print(f"[2/6] Target ready. latent_dim={target.latent_dim}")

    if args.kernel.lower() != "stein":
        raise ValueError("Only the Stein kernel is supported for BLR posterior sampling.")

    print("[3/6] Initializing particles and Stein kernel...")
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, init_key = jax.random.split(rng_key)

    initial_particles = target.sample_initial_particles(
        init_key,
        particle_num=args.particle_num,
        weight_scale=args.weight_init_scale,
        log_alpha_scale=args.log_alpha_init_scale,
    )
    kernel = stein_kernel(sigma=args.bandwidth, score_fn=target.score)
    divergence = build_divergence(args, kernel)
    print(
        "[3/6] Initialization finished. "
        f"particles={args.particle_num}, method={args.method}, optimizer={args.optimizer}, bandwidth={args.bandwidth}"
    )

    print(f"[4/6] Running {args.optimizer} optimizer for {args.step_num} steps...")
    optimizer_info, trajectory = run_particle_optimizer(divergence, rng_key, initial_particles, args)
    final_particles = trajectory[-1]
    print("[4/6] Optimization finished.")
    final_particles_np = np.asarray(final_particles)
    if not np.isfinite(final_particles_np).all():
        print("[4/6] Warning: non-finite particle values detected in final particles.")
    if optimizer_info:
        print(f"[4/6] Optimizer info: {optimizer_info}")

    print("[5/6] Saving particle trajectory and computing metric curves...")
    jnp.save(f"{args.save_path}/trajectory.npy", trajectory)
    jnp.save(f"{args.save_path}/initial_particles.npy", initial_particles)
    jnp.save(f"{args.save_path}/final_particles.npy", final_particles)

    def metric_fn(particles):
        train_metrics = evaluate_posterior_particles(particles, X_train, y_train)
        test_metrics = evaluate_posterior_particles(particles, X_test, y_test)
        return {
            "train_mean_log_likelihood": train_metrics["mean_log_likelihood"],
            "test_mean_log_likelihood": test_metrics["mean_log_likelihood"],
            "train_averaged_accuracy": train_metrics["averaged_accuracy"],
            "test_averaged_accuracy": test_metrics["averaged_accuracy"],
            "mean_alpha": test_metrics["mean_alpha"],
            "finite_fraction": test_metrics["finite_fraction"],
        }

    save_metric_curves(
        trajectory,
        metric_fn,
        args.save_path,
        eval_stride=args.metric_eval_stride,
    )
    print("[5/6] Trajectory artifacts saved.")

    print("[6/6] Evaluating final posterior particles...")
    train_metrics = evaluate_posterior_particles(final_particles, X_train, y_train)
    test_metrics = evaluate_posterior_particles(final_particles, X_test, y_test)
    results = {
        "task": "blr_posterior_sampling",
        "dataset": args.dataset,
        "dataset_metadata": metadata,
        "posterior_model": {
            "latent_state": "[w, log_alpha]",
            "alpha_shape": args.alpha_shape,
            "alpha_rate": args.alpha_rate,
            "latent_dim": int(final_particles.shape[1]),
        },
        "optimizer": {
            "name": args.optimizer,
            "step_num": args.step_num,
            "step_size": args.step_size,
            "info": optimizer_info,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    with open(f"{args.save_path}/metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("[6/6] Evaluation finished. Writing metrics.json")
    print(json.dumps(results, indent=2))


def main(args):
    run_blr_flow(args)


if __name__ == "__main__":
    args = create_dir(get_config())
    print("Program started!")
    print(vars(args))
    main(args)
    print("Program finished!")

    completed_path = f"{args.save_path}__complete"
    if os.path.exists(completed_path):
        shutil.rmtree(completed_path)
    os.rename(args.save_path, completed_path)
    print(f"Job completed. Renamed folder to: {completed_path}")
