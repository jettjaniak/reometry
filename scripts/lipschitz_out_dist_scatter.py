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

dist_all_mask = dist_all[:, -1] > 0.01
print(f"{dist_all_mask.sum().item()=}")
dist = dist_all[dist_all_mask]
dist_norm = dist / dist[:, -1:]
diff = dist_norm[:, 1:] - dist_norm[:, :-1]
start_k = diff[:, :10].mean(dim=1)
max_k = diff.max(dim=1).values

lipschitz_ratio = max_k / start_k
out_dist = dist[:, -1] - dist[:, 0]
in_dist = torch.norm(id.resid_write_a - id.resid_write_b, dim=1)
plt.figure(figsize=(5, 3.5))
plt.scatter(in_dist, lipschitz_ratio, s=1, alpha=0.1)
plt.xlabel("input distance")
plt.ylabel("Lipschitz ratio")
plt.yscale("log")
plt.title(f"model={id.model_name}")

input_path = Path(args.file_path)
output_path = f"plots/lipschitz_out_dist_scatter_{input_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")

mean_in_dist = in_dist.mean().item()
mask = lipschitz_ratio < 1.5  # & (in_dist < mean_in_dist / 3)
print(f"{mask.sum().item()=}")
