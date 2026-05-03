import matplotlib 
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm

import jax
import jax.numpy as jnp
import numpy as np
np.random.seed(49)
import os
import ot
# import ott
# from ott.geometry import costs, pointcloud
# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn
# from densratio import densratio
from pathlib import Path

from kwgflows.divergences.mmd import *

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 20
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

plt.rc('font', size=20)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=18, frameon=False)
plt.rc('xtick', labelsize=14, direction='in')
plt.rc('ytick', labelsize=14, direction='in')
plt.rc('figure', figsize=(6, 4))

FLOW_LIST = ['mmd', 'drmmd']

# def compute_wasserstein_distance_numpy(X, Y):
#     a, b = jnp.ones((X.shape[0], )) / X.shape[0], jnp.ones((Y.shape[0], )) / Y.shape[0]
#     M = ot.dist(X, Y, 'euclidean')
#     W = ot.emd(a, b, M)
#     Wd = (W * M).sum()
#     return Wd

def compute_wasserstein_distance_numpy(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)

    a = np.ones((X.shape[0],), dtype=np.float64) / X.shape[0]
    b = np.ones((Y.shape[0],), dtype=np.float64) / Y.shape[0]

    M = ot.dist(X, Y, metric="euclidean")

    # 更直接：emd2 直接返回 <W, M> 这个标量 cost
    Wd = ot.emd2(a, b, M)
    return Wd

# def compute_wasserstein_distance_numpy(X, Y):
#     """
#     Return Wasserstein-2 distance (W2), not W2^2.
#     """
#     X = np.asarray(X, dtype=np.float64)
#     Y = np.asarray(Y, dtype=np.float64)

#     a = np.ones((X.shape[0],), dtype=np.float64) / X.shape[0]
#     b = np.ones((Y.shape[0],), dtype=np.float64) / Y.shape[0]

#     # W2 要用 squared euclidean
#     M = ot.dist(X, Y, metric="sqeuclidean")

#     # emd2 返回的是最优传输 cost = W2^2
#     W2_sq = ot.emd2(a, b, M)

#     # 返回 W2
#     W2 = np.sqrt(max(W2_sq, 0.0))
#     return W2


# def evaluate(args, ret, rate):
#     eval_freq = int(rate)
#     Path(args.save_path).mkdir(parents=True, exist_ok=True)

#     # --- 选择 CPU device ---
#     cpu = jax.devices("cpu")[0]

#     # --- 抽样索引 ---
#     T = int(ret.Ys.shape[0])
#     eval_idx = np.arange(0, T, eval_freq)

#     # --- 把 evaluation 用到的数据显式搬到 CPU ---
#     # 只搬采样后的 Ys，避免把整条轨迹全拷到 CPU
#     Ys_eval_cpu = jax.device_put(ret.Ys[::eval_freq, :], cpu)  # (Teval, N, d)
#     X_cpu = jax.device_put(ret.divergence.X, cpu)              # (N, d)
#     Y0_cpu = jax.device_put(ret.Ys[0, :], cpu)                 # (N, d)

#     # 保存采样轨迹（保存/画图用 numpy）
#     Ys_eval_np = np.asarray(Ys_eval_cpu)
#     np.save(f"{args.save_path}/Ys.npy", Ys_eval_np)

#     # --- Wasserstein（你用的是 POT/ot，本来就是 CPU）---
#     X_np = np.asarray(X_cpu)
#     wass_distance = []
#     for k in range(Ys_eval_np.shape[0]):
#         wass_distance.append(compute_wasserstein_distance_numpy(Ys_eval_np[k], X_np))
#     wass_distance = np.asarray(wass_distance)

#     # --- MMD / SRMMD 强制在 CPU 上计算 ---
#     with jax.default_device(cpu):
#         # MMD
#         mmd_div = mmd_fixed_target(args, args.kernel_fn, None)
#         mmd_div.pre_compute(X_cpu, Y0_cpu, args.lmbda)

#         # 把“单个时间点”的计算 jit 一下（会在 CPU 编译）
#         mmd_one = jax.jit(lambda Y: mmd_div(Y))
#         mmd_vals = []
#         for k in range(Ys_eval_cpu.shape[0]):
#             mmd_vals.append(mmd_one(Ys_eval_cpu[k]))
#         mmd_vals = jnp.stack(mmd_vals, axis=0)
#         mmd_distance = np.asarray(jnp.sqrt(mmd_vals))

#         # SRMMD
#         srmmd_div = srmmd_fixed_target(args, args.kernel_fn, None)
#         srmmd_div.pre_compute(X_cpu, Y0_cpu, args.lmbda)

#         srmmd_one = jax.jit(lambda Y: srmmd_div(Y))
#         srmmd_vals = []
#         for k in range(Ys_eval_cpu.shape[0]):
#             srmmd_vals.append(srmmd_one(Ys_eval_cpu[k]))
#         srmmd_vals = jnp.stack(srmmd_vals, axis=0)
#         srmmd_distance = np.asarray(srmmd_vals)

#         # 确保 CPU 计算完成再往下走（可选但推荐）
#         jax.block_until_ready(mmd_vals)
#         jax.block_until_ready(srmmd_vals)

#     # --- 横轴：真实迭代次数 ---
#     iters = eval_idx[:len(wass_distance)]

#     # --- 画图/保存（这部分本来就是 CPU）---
#     plt.rcParams.update({
#         "axes.spines.right": False,
#         "axes.spines.top": False,
#         "figure.dpi": 150,
#         "savefig.dpi": 300,
#         "font.size": 11,
#     })

#     fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, constrained_layout=True)
#     axs = axs.ravel()

#     axs[0].plot(iters, wass_distance, linewidth=2)
#     axs[0].set_title("Wasserstein-2")
#     axs[0].set_ylabel("Distance (log)")
#     axs[0].set_yscale("log")
#     axs[0].grid(True, axis="y", alpha=0.3)
#     axs[0].legend(["Wass 2"], loc="upper right", frameon=True)

#     axs[1].plot(iters, mmd_distance, linewidth=2)
#     axs[1].set_title("MMD")
#     axs[1].set_yscale("log")
#     axs[1].grid(True, axis="y", alpha=0.3)
#     axs[1].legend(["MMD"], loc="upper right", frameon=True)

#     axs[2].plot(iters, srmmd_distance, linewidth=2)
#     axs[2].set_title("SRMMD")
#     axs[2].set_yscale("log")
#     axs[2].grid(True, axis="y", alpha=0.3)
#     axs[2].legend(["SRMMD"], loc="upper right", frameon=True)

#     for ax in axs:
#         ax.set_xlabel("Iteration")
#         ax.margins(x=0)

#     out_path = Path(args.save_path) / "distance.png"
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)

#     np.save(f"{args.save_path}/wass_distance.npy", wass_distance)
#     np.save(f"{args.save_path}/mmd_distance.npy", mmd_distance)
#     # np.save(f"{args.save_path}/srmmd_distance.npy", srmmd_distance)
#     return

def evaluate(args, ret, rate):
    eval_freq = int(rate)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    cpu = jax.devices("cpu")[0]

    T = int(ret.Ys.shape[0])
    eval_idx = np.arange(0, T, eval_freq)

    # 只取评估点
    Ys_eval_cpu = jax.device_put(ret.Ys[::eval_freq, :], cpu)
    X_cpu = jax.device_put(ret.divergence.X, cpu)
    Y0_cpu = jax.device_put(ret.Ys[0, :], cpu)

    Ys_eval_np = np.asarray(Ys_eval_cpu)
    X_np = np.asarray(X_cpu)

    # 保存采样轨迹
    np.save(f"{args.save_path}/Ys.npy", Ys_eval_np)

    # -------------------------
    # Wasserstein-2 trajectory
    # -------------------------
    wass_distance = []
    for k in range(Ys_eval_np.shape[0]):
        w2 = compute_wasserstein_distance_numpy(Ys_eval_np[k], X_np)
        wass_distance.append(w2)
    wass_distance = np.asarray(wass_distance, dtype=np.float64)

    # -------------------------
    # MMD / SRMMD on CPU
    # -------------------------
    with jax.default_device(cpu):
        # MMD
        mmd_div = mmd_fixed_target(args, args.kernel_fn, None)
        mmd_div.pre_compute(X_cpu, Y0_cpu, args.lmbda)

        mmd_one = jax.jit(lambda Y: mmd_div(Y))
        mmd_vals = []
        for k in range(Ys_eval_cpu.shape[0]):
            mmd_vals.append(mmd_one(Ys_eval_cpu[k]))
        mmd_vals = jnp.stack(mmd_vals, axis=0)
        jax.block_until_ready(mmd_vals)

        # 避免数值误差导致 sqrt 负数
        mmd_distance = np.sqrt(np.maximum(np.asarray(mmd_vals), 0.0))

        # SRMMD
        srmmd_div = srmmd_fixed_target(args, args.kernel_fn, None)
        srmmd_div.pre_compute(X_cpu, Y0_cpu, args.lmbda)

        srmmd_one = jax.jit(lambda Y: srmmd_div(Y))
        srmmd_vals = []
        for k in range(Ys_eval_cpu.shape[0]):
            srmmd_vals.append(srmmd_one(Ys_eval_cpu[k]))
        srmmd_vals = jnp.stack(srmmd_vals, axis=0)
        jax.block_until_ready(srmmd_vals)

        srmmd_distance = np.asarray(srmmd_vals)

    # -------------------------
    # 最终结果（最后一个时间点）
    # -------------------------
    final_w2 = float(wass_distance[-1])
    final_mmd = float(mmd_distance[-1])
    final_srmmd = float(srmmd_distance[-1])

    print("=" * 60)
    print(f"Final W2   : {final_w2:.10f}")
    print(f"Final MMD  : {final_mmd:.10f}")
    print(f"Final SRMMD : {final_srmmd:.10f}")
    print("=" * 60)

    # 保存最终结果到文本
    with open(f"{args.save_path}/final_metrics.txt", "w") as f:
        f.write(f"Final W2   : {final_w2:.10f}\n")
        f.write(f"Final MMD  : {final_mmd:.10f}\n")
        f.write(f"Final SRMMD : {final_srmmd:.10f}\n")

    # 保存最终结果到 numpy
    np.save(f"{args.save_path}/final_metrics.npy", {
        "final_w2": final_w2,
        "final_mmd": final_mmd,
        "final_srmmd": final_srmmd,
    })

    # 保存整条轨迹
    np.save(f"{args.save_path}/wass_distance.npy", wass_distance)
    np.save(f"{args.save_path}/mmd_distance.npy", mmd_distance)
    np.save(f"{args.save_path}/srmmd_distance.npy", srmmd_distance)

    # -------------------------
    # 画图
    # -------------------------
    iters = eval_idx[:len(wass_distance)]

    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
    })

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, constrained_layout=True)
    axs = axs.ravel()

    axs[0].plot(iters, wass_distance, linewidth=2)
    axs[0].set_title("Wasserstein-2")
    axs[0].set_ylabel("Distance (log)")
    axs[0].set_yscale("log")
    axs[0].grid(True, axis="y", alpha=0.3)
    axs[0].legend(["W2"], loc="upper right", frameon=True)

    axs[1].plot(iters, mmd_distance, linewidth=2)
    axs[1].set_title("MMD")
    axs[1].set_yscale("log")
    axs[1].grid(True, axis="y", alpha=0.3)
    axs[1].legend(["MMD"], loc="upper right", frameon=True)

    axs[2].plot(iters, srmmd_distance, linewidth=2)
    axs[2].set_title("SRMMD")
    axs[2].set_yscale("log")
    axs[2].grid(True, axis="y", alpha=0.3)
    axs[2].legend(["SRMMD"], loc="upper right", frameon=True)

    for ax in axs:
        ax.set_xlabel("Iteration")
        ax.margins(x=0)

    out_path = Path(args.save_path) / "distance.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return {
        "final_w2": final_w2,
        "final_mmd": final_mmd,
        "final_srmmd": final_srmmd,
        "wass_distance": wass_distance,
        "mmd_distance": mmd_distance,
        "srmmd_distance": srmmd_distance,
    }

# def save_animation_1d(args, ret, rate, save_path):
#     num_timesteps = ret.Ys.shape[0]
#     num_frames = max(num_timesteps // rate, 1)
#     num_timesteps_grid = np.arange(1, num_timesteps+1, rate)

#     # Combined update function for both animations
#     def update(frame):
#         # Update for density ratio
#         densratio_obj = densratio(np.array(ret.get_Yt(frame * rate)), X, alpha=alpha, verbose=0)
#         _animate_density_ratio.set_xdata(grid)
#         _animate_density_ratio.set_ydata(densratio_obj.compute_density_ratio(grid))
        
#         # Update for scatter plot
#         x = jnp.clip(ret.get_Yt(frame * rate), -1, 1)
#         y = ret.get_Yt(frame * rate) * 0.0
#         data = np.concatenate((x, y), axis=-1)
#         _animate_scatter.set_offsets(data)
        
#         _animate_distance.set_xdata(num_timesteps_grid[:frame+1])
#         _animate_distance.set_ydata(drmmd_distance[:frame+1])
#         return (_animate_density_ratio, _animate_scatter, _animate_distance)

#     alpha = 0.1
#     X = np.array(ret.divergence.X)
#     densratio_obj = densratio(np.array(ret.get_Yt(0)), X, alpha=alpha, verbose=0)
#     grid = np.linspace(X.min(), X.max(), 100)

#     # Create a single figure with two subplots
#     animate_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     # Initial plot for density ratio on the first subplot
#     _animate_density_ratio, = ax1.plot(grid, densratio_obj.compute_density_ratio(grid), label='Density Ratio')
#     # ax1.set_title(r'Density ratio $\frac{\mu_t}{\pi}$')
#     ax1.set_title(r'Density ratio mu_t/pi$')

#     drmmd_divergence = drmmd_fixed_target(args, args.kernel_fn, None)
#     drmmd_divergence.pre_compute(ret.divergence.X)
#     drmmd_distance = jax.vmap(drmmd_divergence)(ret.Ys[::rate, :])
#     drmmd_distance = np.array(drmmd_distance)
#     # ax2.set_title(r'$\text{DrMMD}(\mu_t \| \pi)$')
#     ax2.set_title(r'DrMMD(mu_t || pi$)')
#     ax2.set_xlabel('Iteration')
#     ax2.set_xlim([0, num_timesteps])
#     ax2.set_ylim([0.0, 1.3])
#     _animate_distance, = ax2.plot(num_timesteps_grid[0], drmmd_distance[0], label='drmmd')

#     # ax3.scatter(ret.divergence.X[:, 0], ret.divergence.X[:, 0] * 0.0, label=r'$\pi$')
#     ax3.scatter(ret.divergence.X[:, 0], ret.divergence.X[:, 0] * 0.0, label='pi')
#     # _animate_scatter = ax3.scatter(jnp.clip(ret.get_Yt(0)[:, 0], -1, 1), ret.get_Yt(0)[:, 0] * 0.0, label=r'$\mu_t$')
#     _animate_scatter = ax3.scatter(jnp.clip(ret.get_Yt(0)[:, 0], -1, 1), ret.get_Yt(0)[:, 0] * 0.0, label='mu_t')
#     ax3.set_xlim(-1.5, 1.5)
#     ax3.axis("off")
#     ax3.legend()

#     # Create a single FuncAnimation for both updates
#     ani_combined = FuncAnimation(
#         animate_fig,
#         update,
#         frames=num_frames,
#         blit=True,
#         interval=50,
#     )

#     ani_combined.save(f'{save_path}/animation_combined.mp4', writer='ffmpeg', fps=1)
#     return

def save_animation_2d(args, ret, rate, save_path):
    num_timesteps = ret.Ys.shape[0]
    num_frames = max(num_timesteps // rate, 1)

    def update(frame):
        _animate_scatter.set_offsets(ret.get_Yt(frame * rate)[:, ::-1])
        return (_animate_scatter,)

    # create initial plot
    animate_fig, animate_ax = plt.subplots()
    # animate_fig.patch.set_alpha(0.)
    # plt.axis('off')
    # animate_ax.scatter(ret.Ys[:, 0], ret.Ys[:, 1], label='source')
    if args.dataset == 'ThreeRing':
        animate_ax.set_xlim(-2.0, 1.0)
        animate_ax.set_ylim(-1.0, 1.0)

    # awkard way to share state for now
    animate_ax.scatter(ret.divergence.X[:, 1], ret.divergence.X[:, 0], label='target')
    _animate_scatter = animate_ax.scatter(ret.get_Yt(0)[:, 1], ret.get_Yt(0)[:, 0], label='target')

    ani_kale = FuncAnimation(
        animate_fig,
        update,
        frames=num_frames,
        # init_func=init,
        blit=True,
        interval=50,
    )
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=rate, bitrate=1800)
        out = os.path.join(save_path, "animation.mp4")
        ani_kale.save(out, writer=writer, dpi=200)
        print(f"[OK] Saved MP4 via ffmpeg -> {out}")
    except Exception as e:
        # ffmpeg 不可用時，退回 GIF（PillowWriter）
        from matplotlib.animation import PillowWriter
        out = os.path.join(save_path, "animation.gif")
        ani_kale.save(out, writer=PillowWriter(fps=rate), dpi=200)
        print(f"[WARN] FFmpeg unavailable ({e}); saved GIF instead -> {out}")
    # ani_kale.save(f'{save_path}/animation.mp4',
                #    writer='ffmpeg', fps=20)
    return    


def save_snapshots_2d(args, ret, save_path, snapshot_every=50, 
    snapshot_iters=None,
    filename="snapshots_with_ground_truth.pdf",
    point_size=14.0,
    alpha=0.32,
):
    """
    Save a horizontal panel of 2D particle snapshots:
    [Iter 0] [Iter 50] [Iter 100] ... [Ground Truth]

    Notes
    -----
    - Each subplot (except the last one) only shows particles at that iteration.
    - The last subplot shows only the ground truth.
    - All subplots share the same axis range for fair comparison.
    """

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Global style: paper-like
    # -------------------------

    font_path = ""
    fm.fontManager.addfont(font_path)
    font_family = "Times New Roman"

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 80,
        "axes.titlesize": 20,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # better editable text in pdf
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })

    num_timesteps = int(ret.Ys.shape[0])
    gt = np.asarray(ret.divergence.X)

    # -------------------------
    # Choose snapshot iterations
    # -------------------------
    if snapshot_iters is None:
        snapshot_every = int(snapshot_every)
        snapshot_iters = list(range(0, num_timesteps, snapshot_every))
        if (num_timesteps - 1) not in snapshot_iters:
            snapshot_iters.append(num_timesteps - 1)
    else:
        snapshot_iters = sorted(
            set(int(t) for t in snapshot_iters if 0 <= int(t) < num_timesteps)
        )

    if len(snapshot_iters) == 0:
        print("[WARN] No valid snapshot iterations.")
        return

    # -------------------------
    # Shared axis limits
    # -------------------------
    Ys_np = np.asarray(ret.Ys)   # (T, N, 2)
    all_pts = np.concatenate([Ys_np.reshape(-1, 2), gt], axis=0)

    # Keep the original convention:
    # x <- [:, 1], y <- [:, 0]
    x_all = all_pts[:, 1]
    y_all = all_pts[:, 0]

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = 0.06 * (x_range + 1e-8)
    y_pad = 0.06 * (y_range + 1e-8)

    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    if getattr(args, "dataset", None) == "ThreeRing":
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.0, 1.0

    # -------------------------
    # Layout
    # -------------------------
    n_snap = len(snapshot_iters)
    ncols = n_snap + 1  # last one for GT

    # more compact and publication-friendly
    fig_w = 2.15 * ncols
    fig_h = 2.25
    fig, axs = plt.subplots(1, ncols, figsize=(fig_w, fig_h))

    if ncols == 1:
        axs = [axs]

    # Paper-friendly muted palette
    traj_color = "#4E79A7"   # muted blue
    gt_color = "#C44E52"     # muted red
    spine_color = "#BBBBBB"

    # -------------------------
    # Snapshot panels
    # -------------------------
    for i, t in enumerate(snapshot_iters):
        ax = axs[i]
        yt = np.asarray(ret.get_Yt(t))

        ax.scatter(
            yt[:, 1],
            yt[:, 0],
            s=point_size,
            alpha=alpha,
            c=traj_color,
            edgecolors="none",
            rasterized=True,   # smaller PDF for many points
        )

        ax.set_title(f"Iter {t}", pad=4, fontweight="regular")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        # cleaner paper-style axes
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        ax.set_facecolor("white")

    # -------------------------
    # Ground truth panel
    # -------------------------
    ax = axs[-1]
    ax.scatter(
        gt[:, 1],
        gt[:, 0],
        s=point_size,
        alpha=alpha,
        c=gt_color,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title("Ground Truth", pad=4, fontweight="regular")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    ax.set_facecolor("white")

    # Tight and neat spacing
    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        bottom=0.06,
        top=0.86,
        wspace=0.08,
    )

    out_file = save_path / filename
    fig.savefig(
        out_file,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] Saved snapshot grid -> {out_file}")