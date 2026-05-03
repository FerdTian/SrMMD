import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import ot


def ksd_distance_empirical_vs_target(Y, stein_kernel):
    """
    Compute KSD between the empirical measure supported on Y and the target
    distribution encoded in `stein_kernel`.

    For a Stein kernel k_pi, KSD^2(emp(Y), pi) = E_{y,y'~emp(Y)}[k_pi(y,y')].
    """
    if not getattr(stein_kernel, "is_stein", False):
        raise ValueError("ksd_distance_empirical_vs_target requires a Stein kernel.")

    Kyy = stein_kernel.make_distance_matrix(Y, Y)
    ksd2 = jnp.maximum(Kyy.mean(), 0.0)
    return jnp.sqrt(ksd2)


def _kernel_mean_yy(kernel, Y):
    Kyy = kernel.make_distance_matrix(Y, Y).mean()
    return Kyy


def _kernel_mean_xx_target(distribution, kernel, rng_key, mc_N=512):
    if hasattr(distribution, "mean_mean_embedding"):
        try:
            return distribution.mean_mean_embedding()
        except TypeError:
            pass

    X = distribution.sample(mc_N, rng_key)
    return kernel.make_distance_matrix(X, X).mean()


def _kernel_mean_x_y_target(distribution, Y):
    kxy_vec = distribution.mean_embedding(Y)
    return kxy_vec.mean()


def mmd_distance_empirical_vs_target(Y, distribution, kernel, rng_key, mc_N=512):
    """
    Standard kernel MMD between emp(Y) and the target distribution.
    This function is intentionally restricted to non-Stein kernels so that MMD
    and KSD semantics do not get mixed.
    """
    if getattr(kernel, "is_stein", False):
        raise ValueError(
            "mmd_distance_empirical_vs_target received a Stein kernel. "
            "Use ksd_distance_empirical_vs_target instead."
        )

    kxx = _kernel_mean_xx_target(distribution, kernel, rng_key, mc_N=mc_N)
    kxy = _kernel_mean_x_y_target(distribution, Y)
    kyy = _kernel_mean_yy(kernel, Y)
    mmd2 = jnp.maximum(kxx + kyy - 2.0 * kxy, 0.0)
    return jnp.sqrt(mmd2)



def wass2_distance_empirical_vs_target_fixedX(Y, X_ref_np):
    """
    W2(emp(Y), emp(X_ref)) using POT EMD2, where X_ref is fixed across time.
    """
    Y_np = np.asarray(Y)
    N = Y_np.shape[0]
    assert X_ref_np.shape[0] == N

    a = np.ones(N, dtype=np.float64) / N
    b = np.ones(N, dtype=np.float64) / N
    M = ot.dist(Y_np, X_ref_np, metric="sqeuclidean").astype(np.float64)
    w2_sq = ot.emd2(a, b, M)
    return float(np.sqrt(w2_sq))
    # X = np.asarray(X_ref_np, dtype=np.float64)
    # Y_np = np.asarray(Y, dtype=np.float64)

    # a = np.ones((X.shape[0],), dtype=np.float64) / X.shape[0]
    # b = np.ones((Y_np.shape[0],), dtype=np.float64) / Y_np.shape[0]

    # # W2 要用 squared euclidean
    # M = ot.dist(X, Y_np, metric="sqeuclidean")

    # # emd2 返回的是最优传输 cost = W2^2
    # W2_sq = ot.emd2(a, b, M)

    # # 返回 W2
    # W2 = np.sqrt(max(W2_sq, 0.0))
    # return W2



def evaluate(
    trajectory,
    distribution,
    metric_kernel,
    rng_key,
    save_every=1,
    eval_every=1,
    mmd_mc_N=512,
    title="GF metrics",
    save_path=None,
):
    """
    Evaluate KSD or MMD (depending on metric_kernel) plus W2 along a trajectory.

    Args:
        trajectory: array of shape (T, N, d)
        metric_kernel: kernel used only for evaluation. If it is a Stein kernel,
            KSD is computed; otherwise standard kernel MMD is computed.
        save_every: the real optimizer-step gap between two consecutive saved
            trajectory frames.
        eval_every: evaluate every `eval_every` saved trajectory frames.
    """
    traj = np.asarray(trajectory)
    T, N, _ = traj.shape

    steps = np.arange(T) * save_every
    idxs = np.arange(0, T, eval_every)
    steps_eval = steps[idxs]

    metric_name = "KSD" if getattr(metric_kernel, "is_stein", False) else "MMD"
    metric_vals = []
    w2_vals = []

    _, kref = jax.random.split(rng_key)
    X_ref = distribution.sample(N, kref)
    X_ref_np = np.asarray(X_ref)

    key = rng_key
    for t in tqdm(idxs, desc=f"Computing {metric_name}/W2 over trajectory"):
        key, k1 = jax.random.split(key)
        Yt = jnp.asarray(traj[t])

        if getattr(metric_kernel, "is_stein", False):
            metric_t = ksd_distance_empirical_vs_target(Yt, metric_kernel)
        else:
            metric_t = mmd_distance_empirical_vs_target(
                Yt, distribution, metric_kernel, rng_key=k1, mc_N=mmd_mc_N
            )

        w2_t = wass2_distance_empirical_vs_target_fixedX(Yt, X_ref_np)
        metric_vals.append(float(metric_t))
        w2_vals.append(float(w2_t))

    metric_vals = np.array(metric_vals)
    w2_vals = np.array(w2_vals)

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)

    def _beautify(ax):
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    ax.plot(steps_eval, metric_vals, linewidth=2.2)
    ax.set_title(f"{metric_name} to target")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_name)
    _beautify(ax)

    i_min = int(np.argmin(metric_vals))
    ax.scatter([steps_eval[i_min]], [metric_vals[i_min]], s=35, zorder=3)
    ax.annotate(
        f"min={metric_vals[i_min]:.3g}",
        (steps_eval[i_min], metric_vals[i_min]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.scatter([steps_eval[-1]], [metric_vals[-1]], s=35, zorder=3)
    ax.annotate(
        f"last={metric_vals[-1]:.3g}",
        (steps_eval[-1], metric_vals[-1]),
        textcoords="offset points",
        xytext=(8, -14),
    )

    ax = axes[1]
    ax.plot(steps_eval, w2_vals, linewidth=2.2)
    ax.set_title("W2 to target")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("W2")
    _beautify(ax)

    i_min = int(np.argmin(w2_vals))
    ax.scatter([steps_eval[i_min]], [w2_vals[i_min]], s=35, zorder=3)
    ax.annotate(
        f"min={w2_vals[i_min]:.3g}",
        (steps_eval[i_min], w2_vals[i_min]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.scatter([steps_eval[-1]], [w2_vals[-1]], s=35, zorder=3)
    ax.annotate(
        f"last={w2_vals[-1]:.3g}",
        (steps_eval[-1], w2_vals[-1]),
        textcoords="offset points",
        xytext=(8, -14),
    )

    fig.suptitle(title, y=1.02)

    if save_path is not None:
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, bbox_inches="tight")

        metric_data = np.stack([steps_eval, metric_vals], axis=1)
        w2_data = np.stack([steps_eval, w2_vals], axis=1)

        if metric_name == "KSD":
            np.save(os.path.join(save_dir, "ksd_vs_step.npy"), metric_data)
            # Keep legacy filename for backward compatibility with older scripts.
            np.save(os.path.join(save_dir, "mmd_vs_step.npy"), metric_data)
        else:
            np.save(os.path.join(save_dir, "mmd_vs_step.npy"), metric_data)

        np.save(os.path.join(save_dir, "w2_vs_step.npy"), w2_data)
        print(f"Saved {metric_name} and W2 metrics to {save_dir}")

    return steps_eval, metric_vals, w2_vals
