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

    return parser.parse_args()


args = get_args()

with open(args.file_path, "rb") as f:
    id = pickle.load(f)


# prompt, step
dist = utils.calculate_resid_read_dist(id)
print(f"{dist.shape=}")
dist_norm = dist / dist[:, -1:]

logit_diff = id.logit_diff
print(f"{logit_diff.shape=}")
logit_diff_norm = logit_diff - logit_diff[:, -1:]
logit_diff_norm = logit_diff_norm / logit_diff_norm[:, 0:1]

alphas = torch.linspace(0, 1, id.inter_steps)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
for i in range(dist_norm.shape[0]):
    axs[0].plot(alphas, dist_norm[i], label=f"prompts D{i+1}")
axs[0].set_xlabel("$\\alpha$")
axs[0].set_ylabel("output distance (normalized)")
axs[0].legend()

for i in range(logit_diff_norm.shape[0]):
    tok_a = id.top_toks_a[i].item()
    tok_b = id.top_toks_b[i].item()
    tok_a_str = id.tokenizer.decode(tok_a)
    tok_b_str = id.tokenizer.decode(tok_b)
    axs[1].plot(
        alphas, logit_diff_norm[i], label=f"D{i+1}, '{tok_a_str}' - '{tok_b_str}'"
    )
axs[1].set_xlabel("$\\alpha$")
axs[1].set_ylabel("logit difference (normalized)")
axs[1].legend()

fig.suptitle(f"model={id.model_name}")

input_path = Path(args.file_path)
output_path = f"plots/ex_diff_plot_{input_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
