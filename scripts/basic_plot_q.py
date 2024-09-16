#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--max-y", type=float, default=1.4)

    return parser.parse_args()


args = get_args()

with open(args.file_path, "rb") as f:
    id = pickle.load(f)


# prompt, step
dist_all = utils.calculate_resid_read_dist(id)
print(f"{dist_all.shape=}")

dist_all_last = dist_all[:, -1]
dist_all_mask = dist_all_last > 0.01
print(f"{dist_all_mask.sum().item()=}")
dist = dist_all[dist_all_mask]
dist_norm = dist / dist[:, -1:]

alphas = torch.linspace(0, 1, id.inter_steps)
plt.figure(figsize=(3.5, 2.5), dpi=300)

# Calculate median and quartiles
median = torch.median(dist_norm, dim=0).values
q1 = torch.quantile(dist_norm, 0.25, dim=0)
q3 = torch.quantile(dist_norm, 0.75, dim=0)

# Plot median as solid line and quartiles as dashed lines
plt.plot(alphas, median, label="median")
color = plt.gca().lines[-1].get_color()
plt.plot(alphas, q1, linestyle="--", label="25th & 75th\npercentiles", color=color)
plt.plot(alphas, q3, linestyle="--", color=color)

plt.legend(loc="upper left")
ymin, ymax = plt.ylim()
plt.ylim(ymin, min(ymax, args.max_y))
plt.xlabel("α")
plt.ylabel("output distance (normalized)")
plt.title(f"model={id.model_name}")

input_path = Path(args.file_path)
output_path = f"plots/basic_plot_q_{input_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
