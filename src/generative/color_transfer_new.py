#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Color transfer via DrMMD / MMD / SRMMD gradient flow + Nearest-Neighbor recolor.

Pipeline:
1) Load a source image and a target image (CIFAR-10 / COCO2017 images / CUB-200 via HF).
2) Treat pixel colors as samples in R^3 (RGB in [0,1]).
3) Subsample N particles from source (Y0) and target (X).
4) Run gradient flow to transport source particles: Y0 -> YT (target fixed).
5) Recolor full source image by nearest neighbor assignment:
      for each source pixel color c, find i = argmin_j ||c - Y0_j||,
      set recolored(c) = YT_i
   (No kernel regression.)
6) Log convergence curves over time for:
   - MMD^2 with Gaussian kernel (same bandwidth)
   - W2 estimated on a fixed subsample using POT (exact EMD on squared cost)

Outputs (into --out_dir/<flow>/<timestamp>/):
- source.png
- target.png
- result.png
- convergence.png
- metrics.csv

Example:
  python color_transfer_nn_drmmd.py --out_dir outputs/demo --dataset cifar \
    --flow drmmd --source_idx 7 --target_idx 1007 --resize 256 \
    --particles 4096 --step_num 3000 --step_size 0.1 --bandwidth 0.12 \
    --lmbda 1e-3 --metric_n 512 --log_every 25

Notes:
- If you set --particles very large, kernel methods may become slow (O(N^2)).
- For NN recolor, scikit-learn is optional; a numpy fallback is included.
"""

import argparse
import csv
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import jax.numpy as jnp
from jax import random


import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image

import ot  # POT

# kwgflows / DrMMD repo imports
from kwgflows.gradient_flow import gradient_flow
from kwgflows.divergences.mmd import (
    mmd_fixed_target,
    drmmd_fixed_target,
    drmmd_fixed_target_adaptive,
    srmmd_fixed_target,
    hrmmd_fixed_target
)
from kwgflows.rkhs.kernels import gaussian_kernel, energy_kernel, rff_gaussian_kernel_from_key

# HF dataset for CUB (optional)
from datasets import load_dataset




def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_rgb01(arr01: np.ndarray, path: str) -> None:
    arr01 = np.clip(arr01, 0.0, 1.0)
    img = Image.fromarray((arr01 * 255.0 + 0.5).astype(np.uint8))
    img.save(path)


def pil_to_rgb01(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def ensure_coco_images(coco_root: str, split: str) -> Path:
    """
    Ensure COCO 2017 images are downloaded & extracted.
    Returns the directory that contains JPEG images, e.g. <root>/val2017
    """
    root = Path(coco_root)
    root.mkdir(parents=True, exist_ok=True)

    url_map = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017":   "http://images.cocodataset.org/zips/val2017.zip",
    }
    if split not in url_map:
        raise ValueError(f"split must be one of {list(url_map.keys())}")

    img_dir = root / split
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return img_dir

    zip_path = root / f"{split}.zip"
    if not zip_path.exists():
        print(f"[INFO] Downloading {split} from {url_map[split]} ...")
        urllib.request.urlretrieve(url_map[split], zip_path)

    print(f"[INFO] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    if not img_dir.exists():
        raise RuntimeError(f"Expected extracted folder {img_dir} not found.")
    return img_dir


def load_coco_pair(
    root: str,
    train: bool,
    source_idx: int,
    target_idx: int,
    resize: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (source_img_rgb01, target_img_rgb01) as float32 arrays in [0,1], HxWx3.
    """
    img_dir = ensure_coco_images(root, "train2017" if train else "val2017")
    paths = sorted([p for p in img_dir.glob("*.jpg")])
    if len(paths) == 0:
        raise RuntimeError(f"No jpg images found in {img_dir}")

    def get_pil_by_idx(idx: int) -> Image.Image:
        if idx < 0 or idx >= len(paths):
            raise IndexError(f"idx={idx} out of range, split has {len(paths)} images")
        return Image.open(paths[idx]).convert("RGB")

    src_img = get_pil_by_idx(source_idx)
    tgt_img = get_pil_by_idx(target_idx)

    if resize is not None and resize > 0:
        src_img = src_img.resize((resize, resize), Image.BICUBIC)
        tgt_img = tgt_img.resize((resize, resize), Image.BICUBIC)

    return pil_to_rgb01(src_img), pil_to_rgb01(tgt_img)


def load_cub_pair(
    root: str,
    train: bool,
    source_idx: int,
    target_idx: int,
    resize: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CUB-200 via HuggingFace dataset (cassiekang/cub200_dataset).
    Returns (source_img_rgb01, target_img_rgb01) in [0,1], HxWx3.
    """
    split = "train" if train else "test"
    ds = load_dataset("cassiekang/cub200_dataset", split=split, cache_dir=os.path.join(root, "hf_cache"))

    n = len(ds)
    if not (0 <= source_idx < n and 0 <= target_idx < n):
        raise IndexError(f"Index out of range for split={split}: n={n}")

    src = ds[source_idx]["image"].convert("RGB")
    tgt = ds[target_idx]["image"].convert("RGB")
    if resize is not None and resize > 0:
        src = src.resize((resize, resize), Image.BICUBIC)
        tgt = tgt.resize((resize, resize), Image.BICUBIC)
    return pil_to_rgb01(src), pil_to_rgb01(tgt)


def load_cifar10_pair(
    root: str,
    train: bool,
    source_idx: int,
    target_idx: int,
    resize: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (source_img_rgb01, target_img_rgb01) in [0,1], HxWx3.
    """
    ds = CIFAR10(root=root, train=train, download=True)
    src_img, _ = ds[source_idx]
    tgt_img, _ = ds[target_idx]

    if not isinstance(src_img, Image.Image):
        src_img = to_pil_image(src_img)
    if not isinstance(tgt_img, Image.Image):
        tgt_img = to_pil_image(tgt_img)

    if resize is not None and resize > 0:
        src_img = src_img.resize((resize, resize), Image.BICUBIC)
        tgt_img = tgt_img.resize((resize, resize), Image.BICUBIC)

    return pil_to_rgb01(src_img), pil_to_rgb01(tgt_img)


def sample_rows(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n, x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def gaussian_kernel_matrix_np(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    x: (N,3), y: (M,3) -> (N,M) where K_ij = exp(-||x_i-y_j||^2/(2 sigma^2))
    """
    x2 = np.sum(x * x, axis=1, keepdims=True)          # (N,1)
    y2 = np.sum(y * y, axis=1, keepdims=True).T        # (1,M)
    d2 = x2 + y2 - 2.0 * (x @ y.T)                     # (N,M)
    return np.exp(-0.5 * d2 / (sigma * sigma + 1e-12))


def mmd2_gaussian_np(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    K_xx = gaussian_kernel_matrix_np(X, X, sigma)
    K_yy = gaussian_kernel_matrix_np(Y, Y, sigma)
    K_xy = gaussian_kernel_matrix_np(X, Y, sigma)
    return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())


def w2_pot_np(X: np.ndarray, Y: np.ndarray) -> float:
    """
    2-Wasserstein distance estimated by solving EMD with squared Euclidean cost (uniform weights).
    Returns W2 (not squared).
    """
    n = X.shape[0]
    a = np.ones((n,), dtype=np.float64) / n
    b = np.ones((n,), dtype=np.float64) / n
    M = ot.dist(X.astype(np.float64), Y.astype(np.float64), metric="sqeuclidean")
    cost = ot.emd2(a, b, M)
    return float(np.sqrt(max(cost, 0.0)))


def nn_recolor(
    src_colors: np.ndarray,  # (P,3) all pixels
    Y0: np.ndarray,          # (N,3) source particles (pre-transport)
    YT: np.ndarray,          # (N,3) transported particles
    chunk: int = 200000,
) -> np.ndarray:
    """
    Nearest neighbor assignment:
      idx(p) = argmin_j ||src_colors[p] - Y0[j]||_2
      recolor[p] = YT[idx(p)]

    Uses scikit-learn if available, else numpy chunked brute-force.
    """
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(Y0)
        _, idx = nn.kneighbors(src_colors, return_distance=True)
        idx = idx[:, 0]
        return YT[idx]
    except Exception:
        # Numpy fallback (brute-force), chunked to control memory
        # Complexity: O(P*N). For big images or big N, this can be slow.
        P = src_colors.shape[0]
        out = np.empty((P, 3), dtype=np.float32)
        Y0_f = Y0.astype(np.float32)
        YT_f = YT.astype(np.float32)

        for s in range(0, P, chunk):
            e = min(s + chunk, P)
            C = src_colors[s:e].astype(np.float32)  # (B,3)
            # compute squared distances to all particles: (B,N)
            # d2 = ||C||^2 + ||Y0||^2 - 2 C Y0^T
            c2 = np.sum(C * C, axis=1, keepdims=True)            # (B,1)
            y2 = np.sum(Y0_f * Y0_f, axis=1, keepdims=True).T    # (1,N)
            d2 = c2 + y2 - 2.0 * (C @ Y0_f.T)                    # (B,N)
            idx = np.argmin(d2, axis=1)                          # (B,)
            out[s:e] = YT_f[idx]
        return out


@dataclass
class MetricsLog:
    steps: list
    mmd2: list
    w2: list


def plot_convergence(log: MetricsLog, out_path: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(11.5, 4.2), dpi=140)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(log.steps, log.mmd2, linewidth=2)
    ax1.set_title("Convergence: MMD$^2$")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("MMD$^2$")
    ax1.set_yscale("log")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(log.steps, log.w2, linewidth=2)
    ax2.set_title("Convergence: $W_2$")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("$W_2$")
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("DrMMD/MMD/SRMMD Color Transfer + NN recolor")

    # Dataset / selection
    parser.add_argument("--data_root", type=str, default="./data",
                        help="where datasets will be downloaded/cached")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--source_idx", type=int, default=10, help="index for source image in split")
    parser.add_argument("--target_idx", type=int, default=87, help="index for target image in split")
    parser.add_argument("--resize", type=int, default=256, help="resize to RxR; <=0 keeps original")
    parser.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "coco", "cub"])

    # Flow
    parser.add_argument("--flow", type=str, default="drmmd", choices=["drmmd", "mmd", "srmmd", "hrmmd"])
    parser.add_argument("--kernel", type=str, default="gaussian", choices=["gaussian", "energy", "rff"])
    parser.add_argument("--particles", type=int, default=4096,
                        help="number of sampled color particles from each image (N)")
    parser.add_argument("--step_num", type=int, default=3000)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--bandwidth", type=float, default=0.12, help="Gaussian kernel sigma")
    parser.add_argument("--lmbda", type=float, default=1e-3, help="DrMMD lambda (ignored for plain mmd)")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument("--adaptive_lmbda", action="store_true", help="use drmmd_fixed_target_adaptive")
    parser.add_argument("--nystrom", type=int, default=0, help="Nyström rank (0 = full inverse)")

    # Logging / NN recolor
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metric_n", type=int, default=512, help="subsample size for metrics")
    parser.add_argument("--log_every", type=int, default=25, help="compute metrics every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nn_chunk", type=int, default=200000,
                        help="chunk size for numpy fallback NN (pixels per chunk)")

    args_cli = parser.parse_args()

    # output folder
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args_cli.out_dir, args_cli.flow, stamp)
    ensure_dir(out_dir)

    train = (args_cli.split == "train")
    resize = None if args_cli.resize <= 0 else args_cli.resize

    # Load image pair
    if args_cli.dataset == "cifar":
        src, tgt = load_cifar10_pair(
            root=args_cli.data_root, train=train,
            source_idx=args_cli.source_idx, target_idx=args_cli.target_idx,
            resize=resize,
        )
    elif args_cli.dataset == "coco":
        src, tgt = load_coco_pair(
            root=args_cli.data_root, train=train,
            source_idx=args_cli.source_idx, target_idx=args_cli.target_idx,
            resize=resize,
        )
    elif args_cli.dataset == "cub":
        src, tgt = load_cub_pair(
            root=args_cli.data_root, train=train,
            source_idx=args_cli.source_idx, target_idx=args_cli.target_idx,
            resize=resize,
        )
    else:
        raise ValueError("Unknown dataset")

    # Save source/target
    save_rgb01(src, os.path.join(out_dir, "source.png"))
    save_rgb01(tgt, os.path.join(out_dir, "target.png"))

    H, W, _ = src.shape
    src_colors = src.reshape(-1, 3).astype(np.float32)  # (P,3)
    tgt_colors = tgt.reshape(-1, 3).astype(np.float32)  # (P,3)

    # Sample particles (equal counts)
    N = int(args_cli.particles)
    X = sample_rows(tgt_colors, N, args_cli.seed + 1).astype(np.float32)  # fixed target particles
    Y0 = sample_rows(src_colors, N, args_cli.seed).astype(np.float32)     # initial source particles

    # Fixed subset indices for metric computation
    metric_n = int(min(args_cli.metric_n, N))
    rng = np.random.default_rng(args_cli.seed + 999)
    metric_idx = rng.choice(N, size=metric_n, replace=False)
    X_metric = X[metric_idx]

    # JAX arrays
    X_j = jnp.array(X)
    Y_j = jnp.array(Y0)

    # Kernel
    if args_cli.kernel == "gaussian":
        kernel = gaussian_kernel(float(args_cli.bandwidth))
    elif args_cli.kernel == "energy":
        kernel = energy_kernel(beta=1.0, sigma=float(args_cli.bandwidth), eps=1e-8)
    elif args_cli.kernel == "rff":
        rng_key = random.PRNGKey(args_cli.seed)
        kernel = rff_gaussian_kernel_from_key(
            rng_key, dim=3, num_features=2048, sigma=float(args_cli.bandwidth)
        )
    else:
        raise ValueError("Unknown kernel")

    # Args object expected by repo
    args = SimpleNamespace(
        step_size=float(args_cli.step_size),
        step_num=int(args_cli.step_num),
        lmbda=float(args_cli.lmbda),
        alpha=float(args_cli.alpha),
        adaptive_lmbda=bool(args_cli.adaptive_lmbda),
        save_path=out_dir,
        nystrom=int(args_cli.nystrom),
    )

    # Divergence wiring
    if args_cli.flow == "mmd":
        divergence = mmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X_j, Y_j, args.lmbda)
    elif args_cli.flow == "drmmd":
        if args_cli.adaptive_lmbda:
            divergence = drmmd_fixed_target_adaptive(args, kernel, None)
            divergence.pre_compute(X_j, Y_j, args.lmbda)
        else:
            divergence = drmmd_fixed_target(args, kernel, None)
            divergence.pre_compute(X_j, Y_j, args.lmbda)
    elif args_cli.flow == "srmmd":
        divergence = srmmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X_j, Y_j, args.lmbda)
    elif args_cli.flow == "hrmmd":
        divergence = hrmmd_fixed_target(args, kernel, None)
        divergence.pre_compute(X_j, Y_j, args.lmbda)
    else:
        raise ValueError("Unknown flow")

    # Run gradient flow
    rng_key = random.PRNGKey(int(args_cli.seed))
    ret = gradient_flow(divergence, rng_key, Y_j, args)

    # Convergence logging
    steps, mmd2_vals, w2_vals = [], [], []
    log_every = max(1, int(args_cli.log_every))

    for t in range(0, args.step_num, log_every):
        Yt = np.array(ret.get_Yt(t)).astype(np.float32)
        Y_metric = Yt[metric_idx]
        mmd2 = mmd2_gaussian_np(X_metric, Y_metric, float(args_cli.bandwidth))
        w2 = w2_pot_np(X_metric, Y_metric)
        steps.append(t)
        mmd2_vals.append(mmd2)
        w2_vals.append(w2)

    # Save metrics.csv
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mmd2", "w2"])
        for s, a, b in zip(steps, mmd2_vals, w2_vals):
            w.writerow([s, a, b])

    # Plot convergence
    plot_convergence(MetricsLog(steps=steps, mmd2=mmd2_vals, w2=w2_vals),
                     os.path.join(out_dir, "convergence.png"))

    # Final particles
    YT = np.array(ret.get_Yt(args.step_num - 1)).astype(np.float32)

    # NN recolor (NO kernel regression)
    recolored = nn_recolor(
        src_colors=src_colors,
        Y0=Y0,
        YT=YT,
        chunk=int(args_cli.nn_chunk),
    )
    out_img = recolored.reshape(H, W, 3)
    save_rgb01(out_img, os.path.join(out_dir, "result.png"))

    print("[OK] Outputs written to:", out_dir)
    print("  - source.png / target.png / result.png")
    print("  - convergence.png (left=MMD^2, right=W2)")
    print("  - metrics.csv")
    print("  - recolor: nearest-neighbor assignment (Y0 -> YT)")


if __name__ == "__main__":
    main()