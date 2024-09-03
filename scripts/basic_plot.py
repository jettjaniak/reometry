#!/usr/bin/env python3
import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt

from reometry import utils
from reometry.typing import *
from reometry.utils import InterpolationData


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--n-curves", "-n", type=int, default=99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--opacity", "-o", type=float, default=0.25)
    parser.add_argument("--max-y", type=float, default=1.4)

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

# select n_curves random curves
random.seed(args.seed)
indices = random.sample(range(dist_norm.shape[0]), args.n_curves)
dist_norm_sample = dist_norm[indices]

alphas = torch.linspace(0, 1, id.inter_steps)
plt.figure(figsize=(5, 3.7))
plt.plot(alphas, dist_norm_sample.T, color="blue", alpha=args.opacity)
ymin, ymax = plt.ylim()
plt.ylim(ymin, min(ymax, args.max_y))
plt.xlabel("interpolation $\\alpha$")
plt.ylabel("normalized L2 distance")
plt.title(f"model={id.model_name}\ndistance from clean run after last layer")
plt.savefig("basic_plot.png", bbox_inches="tight")
