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
dist_all = utils.calculate_resid_read_dist(id)
print(f"{dist_all.shape=}")

dist_all_last = dist_all[:, -1]
dist_all_mask = dist_all_last > 0.01
print(f"{dist_all_mask.sum().item()=}")
dist = dist_all[dist_all_mask]
dist_norm = dist / dist[:, -1:]

alphas = torch.linspace(0, 1, id.inter_steps)
plt.figure(figsize=(5, 3.7))
for i in range(dist_norm.shape[0]):
    plt.plot(alphas, dist_norm[i], label=f"prompts S{i+1}")
plt.xlabel("$\\alpha$")
plt.ylabel("output distance (normalized)")
plt.title(f"model={id.model_name}\n")
plt.legend()
input_path = Path(args.file_path)
output_path = f"plots/ex_sim_plot_{input_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
