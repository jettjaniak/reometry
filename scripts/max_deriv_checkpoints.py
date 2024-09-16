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


def file_to_median_size_tokens(file_path: str) -> tuple[float, int | float, int]:
    with open(file_path, "rb") as f:
        id = pickle.load(f)
        dist = utils.calculate_resid_read_dist(id)
        inter_steps = dist.shape[1]
        dist_norm = dist / dist[:, -1:]
        max_deriv = dist_norm.diff(dim=1).max(dim=1).values * (inter_steps - 1)
        median = torch.median(max_deriv, dim=0).values.item()
        # if this is olmo checkpoint, e.g. olmo-1b[step1000-tokens2B]
        model_name = id.model_name.split("[")[0]
        model_family, str_size = model_name.split("-")
        assert model_family == "olmo"
        size = int(str_size[:-1]) if str_size[:-1].isdigit() else float(str_size[:-1])
        if "tokens" not in id.model_name:
            # fully trained model
            if size == 1:
                tokens = 3050
            elif size == 7:
                tokens = 2750
        else:
            tokens = int(id.model_name.split("tokens")[-1][:-2])
        return median, size, tokens * 1_000_000_000


median_by_tokens_by_size = {}
for file_path in args.file_paths:
    median, size, tokens = file_to_median_size_tokens(file_path)
    print(f"{size=} {tokens=}")
    if size not in median_by_tokens_by_size:
        median_by_tokens_by_size[size] = {}
    median_by_tokens_by_size[size][tokens] = median

fig, ax = plt.subplots(
    1,
    1,
    figsize=(4, 2.5),
    dpi=300,
    sharey=True,
    gridspec_kw={"wspace": 0.05},
)

lowest_tokens = 2e9

for size, median_by_tokens in sorted(median_by_tokens_by_size.items()):
    sorted_tokens = sorted(median_by_tokens.keys())
    non_zero_tokens = [t for t in sorted_tokens if t != 0]
    zero_tokens = [t for t in sorted_tokens if t == 0]

    ax.plot(
        [t if t > lowest_tokens else lowest_tokens for t in sorted_tokens],
        [median_by_tokens[k] for k in sorted_tokens],
        label=f"olmo-{size}b",
        marker="o",
    )

ax.legend(loc="lower right", fontsize="small", title="model")
ax.set_ylabel(r"median of $\max_\alpha$ d'(α)/d(1)")
ax.set_xscale("log")
ax.set_xlabel("training tokens")

# Adjust x-axis ticks and labels
xticks = [float(tick) for tick in ax.get_xticks()]
xticklabels = [l.get_text() for l in ax.get_xticklabels()]
while xticks[0] <= lowest_tokens:
    xticks = xticks[1:]
    xticklabels = xticklabels[1:]
xticks = [lowest_tokens] + xticks
xticklabels = ["0"] + xticklabels
ax.set_xticks(xticks, xticklabels)

# Add a vertical line at x=0.1 to indicate the 0 tokens point
# ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)

sizes_str = "_".join(str(key) for key in sorted(median_by_tokens_by_size.keys()))
output_path = f"plots/max_deriv_checkpoints_{sizes_str}.png"
plt.tight_layout()
plt.savefig(output_path)
