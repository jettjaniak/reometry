#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--threshold", "-t", type=float, default=0.7)

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


def find_first_above_threshold(tensor, jump_thresh):
    mask = tensor > jump_thresh
    return mask.to(torch.int).argmax(dim=-1)


jump_point = find_first_above_threshold(last_dist_norm, args.threshold)


mean_dist = mean_dist_all[combined_mask]
min_loc_mean_dist = min_loc_mean_dist_all[combined_mask]

heatmap = torch.zeros(id.inter_steps, id.inter_steps)
for i, j in zip(jump_point, min_loc_mean_dist):
    heatmap[i, j] += 1

# Trim zero rows and columns
non_zero_rows = torch.any(heatmap > 5, dim=1)
non_zero_cols = torch.any(heatmap > 5, dim=0)
y_start_i, y_end_i = torch.where(non_zero_rows)[0][[0, -1]]
x_start_i, x_end_i = torch.where(non_zero_cols)[0][[0, -1]]

# Adjust the heatmap and alphas range
heatmap = heatmap[y_start_i : y_end_i + 1, x_start_i : x_end_i + 1]
alphas = torch.linspace(0, 1, id.inter_steps)
y_start, y_end = alphas[y_start_i], alphas[y_end_i]
x_start, x_end = alphas[x_start_i], alphas[x_end_i]

heatmap /= mean_dist.shape[0]

fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(
    heatmap, origin="lower", cmap="Reds", extent=[x_start, x_end, y_start, y_end]
)

# Calculate best linear fit
x = min_loc_mean_dist.float() / (id.inter_steps - 1)
y = jump_point.float() / (id.inter_steps - 1)
A = torch.vstack([x, torch.ones_like(x)]).T
m, c = torch.linalg.lstsq(A, y[:, None]).solution.squeeze()

# Calculate correlation coefficient
r = torch.corrcoef(torch.stack([x, y]))[0, 1].item()

# Plot best linear fit line
ax.plot(
    [x_start, x_end],
    [m * x_start + c, m * x_end + c],
    color="black",
    linestyle="--",
    label=f"best fit line",
)

# Add colorbar with the same height as the plot
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("normalized count", rotation=270, labelpad=15)

ax.set_ylabel("jump point α")
ax.set_xlabel("α minimizing distance from mean")
ax.set_title("Correlation between jump and minimum distance points")
ax.legend(loc="upper left")

# Add correlation coefficient to bottom right
ax.text(
    0.95,
    0.05,
    f"r = {r:.2f}",
    transform=ax.transAxes,
    horizontalalignment="right",
    verticalalignment="bottom",
)

input_path = Path(args.file_path)
plt.savefig(f"plots/corr_plot_{input_path.stem}.png", bbox_inches="tight")
