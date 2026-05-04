import json
import os

import matplotlib.pyplot as plt
import numpy as np


def save_metric_curves(trajectory, metric_fn, save_path, eval_stride=10):
    eval_stride = max(1, eval_stride)
    sampled_trajectory = np.asarray(trajectory[::eval_stride])
    if sampled_trajectory.size == 0:
        return

    metric_history = []
    for particles in sampled_trajectory:
        metric_history.append(metric_fn(np.asarray(particles)))

    steps = (np.arange(len(metric_history)) * eval_stride).tolist()
    metric_names = list(metric_history[0].keys())
    metric_curves = {name: [float(metrics[name]) for metrics in metric_history] for name in metric_names}

    with open(os.path.join(save_path, "trajectory_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"steps": steps, "metrics": metric_curves}, handle, indent=2)

    np.savez(
        os.path.join(save_path, "trajectory_metrics.npz"),
        steps=np.asarray(steps, dtype=np.int64),
        **{name: np.asarray(values, dtype=np.float64) for name, values in metric_curves.items()},
    )

    csv_metric_names = [
        name
        for name in (
            "train_averaged_accuracy",
            "test_averaged_accuracy",
            "train_mean_log_likelihood",
            "test_mean_log_likelihood",
            "mean_alpha",
        )
        if name in metric_curves
    ]
    if not csv_metric_names:
        csv_metric_names = metric_names

    csv_path = os.path.join(save_path, "trajectory_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(["step"] + csv_metric_names) + "\n")
        for row_idx, step in enumerate(steps):
            row = [str(step)] + [str(metric_curves[name][row_idx]) for name in csv_metric_names]
            handle.write(",".join(row) + "\n")

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(8, 3 * len(metric_names)), constrained_layout=True)
    if len(metric_names) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, metric_names):
        axis.plot(steps, metric_curves[metric_name], linewidth=2.0)
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.3)

    fig.savefig(os.path.join(save_path, "trajectory_metrics.png"), dpi=150)
    plt.close(fig)
