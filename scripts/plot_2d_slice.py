#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from reometry.typing import *
from reometry.utils import Slice2DData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("-c", "--color-scale", type=float, default=0.3)
    return parser.parse_args()


args = get_args()
input_path = Path(args.file_path)
with open(input_path, "rb") as f:
    data: Slice2DData = pickle.load(f)

n_prompts, inter_steps, _ = data.dist_a.shape
margin = 0.5 * (data.range[1] - data.range[0]) / (inter_steps - 1)
extent_x = [data.range[0] - margin, data.range[1] + margin]
extent_y = list(reversed(extent_x))


def get_colors(idx):
    inside_color = [0.0, 0.0, 0.0]
    inside_color[idx] = 1.0
    for i in range(3):
        if i == idx:
            continue
        inside_color[i] = args.color_scale
    outside_color = [0.0, 0.0, 0.0]
    outside_color[idx] = 1.0 - args.color_scale
    return tuple(inside_color), tuple(outside_color)


def plot_2d_slice(ax, extent, dist_a, dist_b, dist_c, t):
    dist = torch.stack([dist_a, dist_b, dist_c], dim=-1)
    dist /= dist.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values
    ax.imshow(
        1 - dist,
        origin="lower",
        extent=extent,
    )
    s = 180
    lw = 2
    inside_color_a, outside_color_a = get_colors(0)
    inside_color_b, outside_color_b = get_colors(1)
    inside_color_c, outside_color_c = get_colors(2)
    ax.scatter(
        [0],
        [0],
        c=inside_color_a,
        s=s,
        edgecolors=outside_color_a,
        linewidth=lw,
        label="A",
    )
    ax.scatter(
        [1],
        [0],
        c=inside_color_b,
        s=s,
        edgecolors=outside_color_b,
        linewidth=lw,
        label="B",
    )
    ax.scatter(
        [t],
        [1],
        c=inside_color_c,
        s=s,
        edgecolors=outside_color_c,
        linewidth=lw,
        label="C",
    )
    # ax.legend(fontsize="small")
    ax.set_xlabel("α")
    ax.set_ylabel("β", rotation=0, ha="right")
    # ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    # ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])


output_dir = f"plots/slice_2d/{input_path.stem}"
os.makedirs(output_dir, exist_ok=True)

for i in range(n_prompts):
    # fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    extent = extent_x + extent_y
    plot_2d_slice(ax, extent, data.dist_a[i], data.dist_b[i], data.dist_c[i], data.t[i])
    fig.savefig(f"{output_dir}/{i:02d}.png", bbox_inches="tight")
    plt.close(fig)
