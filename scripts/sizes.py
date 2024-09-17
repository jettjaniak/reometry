#!/usr/bin/env python3
import argparse
import pickle
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
class ProccessedFile:
    dist_median: Float[torch.Tensor, "inter_steps"]
    max_deriv_qs: tuple[float, float, float]
    family: str
    size: int | float


def file_to_median_max_deriv_family_size(
    file_path: str,
) -> ProccessedFile:
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
        # e.g. olmo-1b
        family, str_size = id.model_name.rsplit("-", 1)
        size = int(str_size[:-1]) if str_size[:-1].isdigit() else float(str_size[:-1])
        return ProccessedFile(dist_median, tuple(max_deriv_qs.tolist()), family, size)


inter_steps = None
processed_file_by_size_by_family = {}
for file_path in args.file_paths:
    processed_file = file_to_median_max_deriv_family_size(file_path)
    family = processed_file.family
    size = processed_file.size
    print(f"{family=} {size=}")
    if family not in processed_file_by_size_by_family:
        processed_file_by_size_by_family[family] = {}
    processed_file_by_size_by_family[family][size] = processed_file
    dist_median = processed_file.dist_median
    if inter_steps is None:
        inter_steps = dist_median.shape[0]
    assert dist_median.shape[0] == inter_steps

num_families = len(processed_file_by_size_by_family)
num_axes = num_families + 1
# Create the figure
fig = plt.figure(figsize=(0.1 + 2.85 * num_axes, 2.5), dpi=600)
gs = gridspec.GridSpec(
    1,
    num_axes + 1,
    width_ratios=[1] * num_families + [0.2, 1],
    wspace=0.05,
)  # include space for y label of last plot
# Create subplots
ax_1 = plt.subplot(gs[0])
ax_1.set_ylabel("median d(α)/d(1)")
axes_family_after_first = [plt.subplot(gs[i]) for i in range(1, num_families)]
axes_family = [ax_1] + axes_family_after_first


alphas = torch.linspace(0, 1, inter_steps)
for i, (ax, (family, processed_file_by_size)) in enumerate(
    zip(axes_family, processed_file_by_size_by_family.items())
):
    for size, processed_file in sorted(processed_file_by_size.items()):
        dist_median = processed_file.dist_median
        ax.plot(alphas, dist_median, label=f"{size}B")

    ax.legend(loc="lower right", fontsize="small", title="param.")
    ax.set_xlabel("α")
    ax.set_title(f"family={family}")

    # Simplify x-axis tick labels
    current_labels = ax.get_xticklabels()
    simplified_labels = [
        label.get_text().rstrip("0").rstrip(".") for label in current_labels
    ]
    ax.set_xticklabels(simplified_labels)

ylim_1 = ax_1.get_ylim()
yticks_1 = ax_1.get_yticks()
for ax in axes_family_after_first:
    ax.set_yticks(yticks_1, [""] * len(yticks_1))
    ax.set_ylim(*ylim_1)
ax_last = plt.subplot(gs[-1])


def plot_max_deriv():
    # ax_last.set_xscale("log")
    ax_last.set_xlabel("number of parameters")
    ax_last.set_ylabel(r"median $\text{max}_\alpha\,d'(\alpha)/d(1)$")
    for family, processed_file_by_size in sorted(
        processed_file_by_size_by_family.items()
    ):
        sorted_sizes = sorted(processed_file_by_size.keys())
        q1_deriv, median_deriv, q3_deriv = zip(
            *[processed_file_by_size[k].max_deriv_qs for k in sorted_sizes]
        )
        # ax_last.plot(sorted_sizes, median_deriv, label=f"{family}", marker="o")
        ax_last.errorbar(
            sorted_sizes,
            median_deriv,
            yerr=[
                np.array(median_deriv) - np.array(q1_deriv),
                np.array(q3_deriv) - np.array(median_deriv),
            ],
            label=f"{family}",
            marker="o",
            capsize=3,
        )

    ax_last.legend(loc="upper left", fontsize="small", title="family")

    # Adjust x-axis ticks and labels
    # xticks = [float(tick) for tick in ax_last.get_xticks()]
    # xticklabels = [l.get_text() for l in ax_last.get_xticklabels()]
    # while xticks[0] <= lowest_tokens:
    #     xticks = xticks[1:]
    #     xticklabels = xticklabels[1:]
    # xticks = [lowest_tokens] + xticks
    # xticklabels = ["0"] + xticklabels
    xticks = [0.5, 1, 1.5, 7]
    xticklabels = ["", "1B", "", "7B"]
    ax_last.set_xticks(xticks, xticklabels)
    # ax_last.set_yticks([1, 5, 10, 15], ["1", "5", "10", "15"])
    # xmin, _xmax = ax_last.get_xlim()
    # ax_last.set_xlim(xmin, 1e13)


plot_max_deriv()


# fig, axes = plt.subplots(
#     1,
#     num_families,
#     figsize=(0.1 + 2.85 * num_families, 2.5),
#     dpi=300,
#     sharey=True,
#     gridspec_kw={"wspace": 0.05},
# )
# if num_families == 1:
#     axes = [axes]

# alphas = torch.linspace(0, 1, inter_steps)
# for i, (ax, (family, median_by_size)) in enumerate(
#     zip(axes, median_by_size_by_family.items())
# ):
#     for size, median in sorted(median_by_size.items()):
#         ax.plot(alphas, median, label=f"{size}B")

#     ax.legend(loc="lower right", fontsize="small", title="param.")
#     ax.set_xlabel("α")
#     ax.set_title(f"family={family}")

#     # Simplify x-axis tick labels
#     current_labels = ax.get_xticklabels()
#     simplified_labels = [
#         label.get_text().rstrip("0").rstrip(".") for label in current_labels
#     ]
#     ax.set_xticklabels(simplified_labels)

# # Set y-label only for the leftmost subplot
# axes[0].set_ylabel("median d(α)/d(1)")

output_path = f"plots/sizes_{'_'.join(processed_file_by_size_by_family.keys())}.png"
plt.savefig(output_path, bbox_inches="tight")
