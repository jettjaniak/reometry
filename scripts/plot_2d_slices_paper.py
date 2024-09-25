#!/usr/bin/env python3
import argparse
import os
import pickle
import re
from pathlib import Path

import matplotlib
import torch
from matplotlib import pyplot as plt

from reometry.typing import *
from reometry.utils import Slice2DData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", type=str, nargs="+")
    parser.add_argument("-c", "--color-scale", type=float, default=0.5)
    return parser.parse_args()


def extract_model_info(file_stem):
    match = re.search(r"slice_([a-z0-9-]+)\[step\d+-tokens(\d+)B\].*", file_stem)
    if match:
        model = match.group(1)
        tokens = match.group(2)
        return model, f"{tokens}B"

    # Fallback for files without token information
    match = re.search(r"slice_([a-z0-9-]+)_*.*", file_stem)
    if match:
        model = match.group(1)
        if model == "olmo-7b":
            tokens = "2750"
        elif model == "olmo-1b":
            tokens = "3050"
        else:
            raise ValueError(f"Unknown model without token info: {model}")
        return model, f"{tokens}B"

    raise ValueError(f"Unable to extract model info from filename: {file_stem}")


def get_colors(idx, color_scale):
    inside_color = [0.0, 0.0, 0.0]
    inside_color[idx] = 1.0
    for i in range(3):
        if i == idx:
            continue
        inside_color[i] = color_scale
    outside_color = [0.0, 0.0, 0.0]
    outside_color[idx] = 1.0 - color_scale
    return tuple(inside_color), tuple(outside_color)


def plot_2d_slice(ax, dist_a, dist_b, dist_c, t, color_scale):
    dist = torch.stack([dist_a, dist_b, dist_c], dim=-1)
    dist /= dist.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values
    ax.imshow(
        1 - dist,
        origin="lower",
        extent=extent,
    )
    s = 40
    lw = 1
    inside_color_a, outside_color_a = get_colors(0, color_scale)
    inside_color_b, outside_color_b = get_colors(1, color_scale)
    inside_color_c, outside_color_c = get_colors(2, color_scale)
    ax.scatter(
        [0],
        [0],
        color=inside_color_a,
        s=s,
        edgecolors=outside_color_a,
        linewidth=lw,
    )
    ax.scatter(
        [1],
        [0],
        color=inside_color_b,
        s=s,
        edgecolors=outside_color_b,
        linewidth=lw,
    )
    ax.scatter(
        [t],
        [1],
        color=inside_color_c,
        s=s,
        edgecolors=outside_color_c,
        linewidth=lw,
    )


args = get_args()
num_files = len(args.file_paths)

# Load all data first
all_data = []
for file_path in args.file_paths:
    input_path = Path(file_path)
    with open(input_path, "rb") as f:
        data: Slice2DData = pickle.load(f)
    model, tokens = extract_model_info(input_path.stem)
    all_data.append((model, tokens, data))

n_prompts = all_data[0][2].dist_a.shape[0]

# Use the model name from the first file for the output directory
output_dir = os.path.join("plots", "slice_2d_paper", all_data[0][0])
os.makedirs(output_dir, exist_ok=True)

for idx in range(n_prompts):
    fig, axes = plt.subplots(1, num_files, figsize=(8.65, 2.5), dpi=300, sharey=True)
    if num_files == 1:
        axes = [axes]

    for ax, (model, tokens, data) in zip(axes, all_data):
        inter_steps, _ = data.dist_a[idx].shape
        margin = 0.5 * (data.range[1] - data.range[0]) / (inter_steps - 1)
        extent_x = [data.range[0] - margin, data.range[1] + margin]
        extent_y = list(reversed(extent_x))
        extent = extent_x + extent_y

        plot_2d_slice(
            ax,
            data.dist_a[idx],
            data.dist_b[idx],
            data.dist_c[idx],
            data.t[idx],
            args.color_scale,
        )
        ax.set_title(tokens)

    axes[0].set_ylabel("β", rotation=0, ha="right")
    for ax in axes:
        ax.set_xlabel("α")

    output_path = os.path.join(output_dir, f"slice_2d_{idx:02d}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

print(f"Plots saved in {output_dir}")
