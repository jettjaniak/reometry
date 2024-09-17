#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData, InterpolationDataDifferent


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path_sim", type=str)
    parser.add_argument("file_path_diff", type=str)
    return parser.parse_args()


args = get_args()


def process_sim(file_path: str):
    with open(file_path, "rb") as f:
        id = pickle.load(f)
    assert isinstance(id, InterpolationData)

    dist = utils.calculate_resid_read_dist(id)
    dist_norm = dist / dist[:, -1:]
    return dist_norm, id.model_name


def process_diff(file_path: str):
    with open(file_path, "rb") as f:
        id = pickle.load(f)
    assert isinstance(id, InterpolationDataDifferent)

    dist = utils.calculate_resid_read_dist(id)
    dist_norm = dist / dist[:, -1:]

    logit_diff = id.logit_diff
    logit_diff_norm = logit_diff - logit_diff[:, -1:]
    logit_diff_norm = logit_diff_norm / logit_diff_norm[:, 0:1]

    tokenizer = id.tokenizer
    toks_a = [tokenizer.decode(tok) for tok in id.top_toks_a]
    toks_b = [tokenizer.decode(tok) for tok in id.top_toks_b]

    return dist_norm, logit_diff_norm, toks_a, toks_b, id.model_name


num_axes = 3
# Create the figure
fig = plt.figure(figsize=(0.1 + 2.85 * num_axes, 2.5), dpi=600)
gs = gridspec.GridSpec(
    1,
    num_axes + 1,
    width_ratios=[1, 1, 0.25, 1],
    wspace=0.05,
)  # include space for y label of last plot
# Create subplots
ax_1 = plt.subplot(gs[0])
ax_2 = plt.subplot(gs[1])
ax_3 = plt.subplot(gs[3])


def plot_sim():
    dist, model_name = process_sim(args.file_path_sim)
    alphas = torch.linspace(0, 1, dist.shape[1])
    for prompt_i in range(dist.shape[0]):
        ax_1.plot(alphas, dist[prompt_i], label=f"S{prompt_i+1}")
    ax_1.set_xlabel("$\\alpha$")
    ax_1.legend(loc="upper left", fontsize="small", title="prompts")
    return model_name


def plot_diff():
    dist, logit_diff, toks_a, toks_b, model_name = process_diff(args.file_path_diff)
    alphas = torch.linspace(0, 1, dist.shape[1])
    for prompt_i in range(dist.shape[0]):
        ax_2.plot(alphas, dist[prompt_i], label=f"D{prompt_i+1}")
        ax_3.plot(
            alphas,
            1 - logit_diff[prompt_i],
            label=f"D{prompt_i+1}",
        )
    ax_2.set_xlabel("$\\alpha$")
    ax_3.set_xlabel("$\\alpha$")
    ax_2.legend(loc="upper left", fontsize="small", title="prompts")
    ax_3.legend(loc="upper left", fontsize="small", title="prompts")
    return model_name


model_name_sim = plot_sim()
model_name_diff = plot_diff()
assert model_name_sim == model_name_diff

ax_1.set_ylabel("d(α)/d(1)")
ax_1.set_title(f"similar")
# ylim_1 = ax_1.get_ylim()
# ax_2.set_ylim(*ylim_1)
# yticks_1 = ax_1.get_yticks()
ax_2.set_yticklabels([""] * len(ax_2.get_yticks()))
ax_2.set_title(f"dissimilar")
# ax_2.set_yticklabels([""] * len(yticks_1))
ax_3.set_ylabel("logit diff (normalized)")
ax_3.set_title(f"dissimilar")

output_path = f"plots/examples_{model_name_sim}.png"
plt.savefig(output_path, bbox_inches="tight")
