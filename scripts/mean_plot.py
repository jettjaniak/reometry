#!/usr/bin/env python3
import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--intervals", "-i", type=int, default=20)
    parser.add_argument("--curves-per-interval", "-c", type=int, default=5)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--opacity", "-o", type=float, default=0.5)
    parser.add_argument("--max-y", type=float, default=1.4)

    return parser.parse_args()


args = get_args()

with open(args.file_path, "rb") as f:
    id = pickle.load(f)


# prompt, step
last_dist_all = utils.calculate_resid_read_dist(id)
print(f"{last_dist_all.shape=}")
mean_dist_all = utils.calculate_resid_write_mean_dist(id)
print(f"{mean_dist_all.shape=}")

last_dist_all_last = last_dist_all[:, -1]
last_dist_all_mask = last_dist_all_last > 0.01
print(f"{last_dist_all_mask.sum().item()=}")

mean_dist_all_min_max_diff = (
    mean_dist_all.max(dim=-1).values - mean_dist_all.min(dim=-1).values
)
mean_dist_all_min_max_diff_mask = mean_dist_all_min_max_diff > 3
print(f"{mean_dist_all_min_max_diff_mask.sum().item()=}")

min_loc_mean_dist_all = mean_dist_all.argmin(dim=-1)
min_loc_mean_dist_all_mask = (min_loc_mean_dist_all > 1) & (
    min_loc_mean_dist_all < id.inter_steps - 2
)
print(f"{min_loc_mean_dist_all_mask.sum().item()=}")

combined_mask = (
    last_dist_all_mask & mean_dist_all_min_max_diff_mask & min_loc_mean_dist_all_mask
)
print(f"{combined_mask.sum().item()=}")
last_dist = last_dist_all[combined_mask]
last_dist_norm = last_dist / last_dist[:, -1:]
mean_dist = mean_dist_all[combined_mask]
min_loc_mean_dist = min_loc_mean_dist_all[combined_mask]

alphas = torch.linspace(0, 1, id.inter_steps)
min_mean_alphas = alphas[min_loc_mean_dist]
min_mean_alphas_min, min_mean_alphas_max = min_mean_alphas.min(), min_mean_alphas.max()
min_mean_alphas_ls = torch.linspace(
    min_mean_alphas_min, min_mean_alphas_max, args.intervals
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
random.seed(args.seed)

# Create a colormap
cmap = matplotlib.colormaps["jet"].reversed()
norm = matplotlib.colors.Normalize(vmin=min_mean_alphas_min, vmax=min_mean_alphas_max)

# Create lists to store all curves and their corresponding colors
all_mean_curves = []
all_last_curves = []
all_colors = []

for x, y in zip(min_mean_alphas_ls[:-1], min_mean_alphas_ls[1:]):
    color = cmap(norm((x + y) / 2))
    mask = (min_mean_alphas >= x) & (min_mean_alphas < y)
    mask = mask.squeeze()
    sel_last_dist = last_dist_norm[mask]
    sel_mean_dist = mean_dist[mask]
    # take up to ... random samples
    n_curves = min(args.curves_per_interval, sel_last_dist.shape[0])
    indices = random.sample(range(sel_last_dist.shape[0]), n_curves)
    sel_last_dist = sel_last_dist[indices]
    sel_mean_dist = sel_mean_dist[indices]
    all_mean_curves.extend(sel_mean_dist)
    all_last_curves.extend(sel_last_dist)
    all_colors.extend([color] * n_curves)

# Randomize the order of curves
random_order = random.sample(range(len(all_mean_curves)), len(all_mean_curves))

for i in random_order:
    ax[0].plot(alphas, all_mean_curves[i], color=all_colors[i], alpha=args.opacity)
    ax[1].plot(alphas, all_last_curves[i], color=all_colors[i], alpha=args.opacity)

ax[0].set_title(f"")
ymin, ymax = ax[1].get_ylim()
ax[1].set_ylim(ymin, min(ymax, args.max_y))

ax[0].set_title(f"distance from mean after layer 0")
ax[0].set_ylabel("L2 distance")

ax[1].set_title(f"distance from clean run after last layer")
ax[1].set_ylabel("normalized L2 distance")

for a in ax:
    a.set_xlabel("interpolation $\\alpha$")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), orientation="horizontal", aspect=30)
cbar.set_label("$\\alpha$ minimizing distance from mean")
plt.savefig("mean_plot.png", bbox_inches="tight")
