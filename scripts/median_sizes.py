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
    parser.add_argument("file_paths", type=str, nargs="+")
    return parser.parse_args()


args = get_args()


def file_to_median_family_and_size(file_path: str) -> tuple[torch.Tensor, str, float]:
    with open(file_path, "rb") as f:
        id = pickle.load(f)
        dist = utils.calculate_resid_read_dist(id)
        dist_norm = dist / dist[:, -1:]
        median = torch.median(dist_norm, dim=0).values
        # e.g. olmo-1b
        family, str_size = id.model_name.rsplit("-", 1)
        size = float(str_size[:-1])
        return median, family, size


inter_steps = None
median_by_size_by_family = {}
for file_path in args.file_paths:
    median, family, size = file_to_median_family_and_size(file_path)
    print(f"{family=} {size=}")
    if family not in median_by_size_by_family:
        median_by_size_by_family[family] = {}
    median_by_size_by_family[family][size] = median
    if inter_steps is None:
        inter_steps = median.shape[0]
    assert median.shape[0] == inter_steps

num_families = len(median_by_size_by_family)
fig, axes = plt.subplots(
    1, num_families, figsize=(0.2 + 3.3 * num_families, 2.5), dpi=300, sharey=True
)
if num_families == 1:
    axes = [axes]

alphas = torch.linspace(0, 1, inter_steps)
for ax, (family, median_by_size) in zip(axes, median_by_size_by_family.items()):
    for size, median in sorted(median_by_size.items()):
        ax.plot(alphas, median, label=f"{size}B")

    ax.legend(loc="upper left", fontsize="small", title="param.")
    ax.set_xlabel("α")
    ax.set_title(f"family={family}")

axes[0].set_ylabel("median d(α)/d(1)")

output_path = f"plots/median_sizes_{'_'.join(median_by_size_by_family.keys())}.png"
plt.savefig(output_path, bbox_inches="tight")
