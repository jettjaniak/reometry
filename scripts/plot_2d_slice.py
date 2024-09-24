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
    parser.add_argument("-c", "--color-scale", type=float, default=0.5)
    return parser.parse_args()


args = get_args()
input_path = Path(args.file_path)
with open(input_path, "rb") as f:
    data: Slice2DData = pickle.load(f)

n_prompts, inter_steps, _ = data.dist_a.shape
margin = 0.5 * (data.range[1] - data.range[0]) / (inter_steps - 1)
extent_x = [data.range[0] - margin, data.range[1] + margin]
extent_y = list(reversed(extent_x))
extent = extent_x + extent_y


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


def plot_2d_slice(dist_a, dist_b, dist_c, t, output_path):
    plt.figure(figsize=(4, 4))
    dist = torch.stack([dist_a, dist_b, dist_c], dim=-1)
    dist /= dist.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values
    plt.imshow(
        1 - dist,
        origin="lower",
        extent=extent,
    )
    s = 180
    lw = 2
    inside_color_a, outside_color_a = get_colors(0)
    inside_color_b, outside_color_b = get_colors(1)
    inside_color_c, outside_color_c = get_colors(2)
    plt.scatter(
        [0],
        [0],
        color=inside_color_a,
        s=s,
        edgecolors=outside_color_a,
        linewidth=lw,
        label="A",
    )
    plt.scatter(
        [1],
        [0],
        color=inside_color_b,
        s=s,
        edgecolors=outside_color_b,
        linewidth=lw,
        label="B",
    )
    plt.scatter(
        [t],
        [1],
        color=inside_color_c,
        s=s,
        edgecolors=outside_color_c,
        linewidth=lw,
        label="C",
    )
    plt.axis("off")
    plt.gca().set_frame_on(False)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


output_dir = f"plots/slice_2d/{input_path.stem}"
os.makedirs(output_dir, exist_ok=True)

for i in range(n_prompts):
    plot_2d_slice(
        data.dist_a[i],
        data.dist_b[i],
        data.dist_c[i],
        data.t[i],
        f"{output_dir}/{i:02d}.png",
    )
