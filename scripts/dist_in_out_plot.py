#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--model_name", "-m", type=str)
    parser.add_argument("--num_bins", "-b", type=int, default=20)

    return parser.parse_args()


args = get_args()

with open(args.file_path, "rb") as f:
    dist_in, dist_out = pickle.load(f)


# Assuming dist_in and dist_out are your input and output distances
# If you're loading from a file, keep that part of the code

# Create bins along the x-axis
bin_edges = np.linspace(min(dist_in), max(dist_in), args.num_bins + 1)

# Calculate median and percentiles for each bin
median = binned_statistic(dist_in, dist_out, statistic="median", bins=bin_edges)[0]
percentile_25 = binned_statistic(
    dist_in, dist_out, statistic=lambda x: np.percentile(x, 25), bins=bin_edges
)[0]
percentile_75 = binned_statistic(
    dist_in, dist_out, statistic=lambda x: np.percentile(x, 75), bins=bin_edges
)[0]

# Calculate bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Create the plot
plt.figure(figsize=(5, 3.5))

# Plot original scattered data with low alpha
plt.scatter(dist_in, dist_out, alpha=0.2, s=3, color="lightgray")

# Plot median as solid line
plt.plot(bin_centers, median, label="median")
color = plt.gca().lines[-1].get_color()
# Plot 25th and 75th percentiles as dashed lines
plt.plot(
    bin_centers,
    percentile_25,
    linestyle="--",
    label="25th & 75th\npercentiles",
    color=color,
)
plt.plot(bin_centers, percentile_75, linestyle="--", color=color)

plt.xlabel("input distance")
plt.ylabel("output distance")
plt.title(f"model={args.model_name}")
plt.legend()

input_path = Path(args.file_path)
output_path = f"plots/dist_in_out_{input_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
