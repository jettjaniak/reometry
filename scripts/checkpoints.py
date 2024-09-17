#!/usr/bin/env python3
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from beartype import beartype
from matplotlib import gridspec

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


@beartype
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", type=str, nargs="+")
    return parser.parse_args()


args = get_args()


@beartype
@dataclass
class ProcessedFile:
    dist_median: Float[torch.Tensor, "inter_steps"]
    max_deriv_qs: tuple[float, float, float]
    size: int | float
    tokens: int


@beartype
def file_to_processed_file(file_path: str) -> ProcessedFile:
    with open(file_path, "rb") as f:
        id = pickle.load(f)
        dist = utils.calculate_resid_read_dist(id)
        dist_norm = dist / dist[:, -1:]
        dist_median = torch.median(dist_norm, dim=0).values
        inter_steps = dist.shape[1]
        max_deriv = dist_norm.diff(dim=1).max(dim=1).values * (inter_steps - 1)
        max_deriv_qs = torch.quantile(
            max_deriv, q=torch.tensor([0.25, 0.5, 0.75]), dim=0
        )
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
        return ProcessedFile(dist_median, tuple(max_deriv_qs.tolist()), size, tokens)


inter_steps = None
processed_file_by_tokens_by_size = {}
for file_path in args.file_paths:
    processed_file = file_to_processed_file(file_path)
    size, tokens = processed_file.size, processed_file.tokens
    print(f"{size=} {tokens=}")
    if size not in processed_file_by_tokens_by_size:
        processed_file_by_tokens_by_size[size] = {}
    processed_file_by_tokens_by_size[size][tokens] = processed_file
    if inter_steps is None:
        inter_steps = processed_file.dist_median.shape[0]
    assert processed_file.dist_median.shape[0] == inter_steps

num_sizes = len(processed_file_by_tokens_by_size)
num_axes = num_sizes + 1
# Create the figure
fig = plt.figure(figsize=(0.1 + 2.85 * num_axes, 2.5), dpi=600)
gs = gridspec.GridSpec(
    1,
    num_axes + 1,
    width_ratios=[1] * num_sizes + [0.2, 1],
    wspace=0.05,
)  # include space for y label of last plot
# Create subplots
ax_1 = plt.subplot(gs[0])
ax_1.set_ylabel("median d(α)/d(1)")
axes_size_after_first = [plt.subplot(gs[i]) for i in range(1, num_sizes)]
axes_size = [ax_1] + axes_size_after_first

show_tokens_by_size = {
    1: [0, 4, 10, 31, 3050],
    7: [0, 10, 100, 501, 2750],
}

alphas = torch.linspace(0, 1, inter_steps)
for i, (ax, (size, processed_file_by_tokens)) in enumerate(
    zip(axes_size, processed_file_by_tokens_by_size.items())
):
    for tokens, processed_file in sorted(processed_file_by_tokens.items()):
        if tokens not in show_tokens_by_size[size]:
            continue
        ax.plot(alphas, processed_file.dist_median, label=f"{tokens}B")

    ax.legend(loc="lower right", fontsize="small", title="train tok.")
    ax.set_xlabel("α")
    ax.set_title(f"model=olmo-{size}b")

    # Simplify x-axis tick labels
    current_labels = ax.get_xticklabels()
    simplified_labels = [
        label.get_text().rstrip("0").rstrip(".") for label in current_labels
    ]
    ax.set_xticklabels(simplified_labels)

ylim_1 = ax_1.get_ylim()
yticks_1 = ax_1.get_yticks()
for ax in axes_size_after_first:
    ax.set_yticks(yticks_1, [""] * len(yticks_1))
    ax.set_ylim(*ylim_1)
ax_last = plt.subplot(gs[-1])


def plot_max_deriv():
    lowest_tokens = 1e9
    ax_last.set_xscale("log")
    ax_last.set_xlabel("training tokens")
    ax_last.set_ylabel(r"median $\text{max}_\alpha\,d'(\alpha)/d(1)$")
    for size, processed_file_by_tokens in sorted(
        processed_file_by_tokens_by_size.items()
    ):
        sorted_tokens_b = sorted(processed_file_by_tokens.keys())
        sorted_tokens = [t * 1_000_000_000 for t in sorted_tokens_b]
        x = [t if t > lowest_tokens else lowest_tokens for t in sorted_tokens]
        q1_deriv, median_deriv, q3_deriv = zip(
            *[processed_file_by_tokens[k].max_deriv_qs for k in sorted_tokens_b]
        )
        ax_last.errorbar(
            x,
            median_deriv,
            yerr=[
                np.array(median_deriv) - np.array(q1_deriv),
                np.array(q3_deriv) - np.array(median_deriv),
            ],
            label=f"olmo-{size}b",
            marker="o",
            capsize=3,
        )

    ax_last.set_xlim(lowest_tokens / 2, 1e13)
    xticks = ax_last.get_xticks()
    xticklabels = [l.get_text() for l in ax_last.get_xticklabels()]
    # while xticks[0] <= lowest_tokens:
    #     xticks = xticks[1:]
    #     xticklabels = xticklabels[1:]
    # xticks = [lowest_tokens] + xticks
    # xticklabels = ["0"] + xticklabels
    # ax_last.set_xticks(xticks, xticklabels)
    # # ax_last.set_yticks([1, 5, 10, 15], ["1", "5", "10", "15"])


plot_max_deriv()

sizes_str = "_".join(
    str(key) for key in sorted(processed_file_by_tokens_by_size.keys())
)
output_path = f"plots/checkpoints_{sizes_str}.png"
plt.savefig(output_path, bbox_inches="tight")
