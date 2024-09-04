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
    parser.add_argument("no_arc_file_path", type=str)
    parser.add_argument("--n-curves", "-n", type=int, default=99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--opacity", "-o", type=float, default=0.25)
    parser.add_argument("--max-y", type=float, default=1.4)

    return parser.parse_args()


args = get_args()
no_arc_file_path = Path(args.no_arc_file_path)
with open(no_arc_file_path, "rb") as f:
    id_no_arc = pickle.load(f)

# .../... -> .../arc_...
arc_file_path = (
    no_arc_file_path.parent / f"arc_{no_arc_file_path.stem}{no_arc_file_path.suffix}"
)
with open(arc_file_path, "rb") as f:
    id_arc = pickle.load(f)


# prompt, step
dist_noarc_all = utils.calculate_resid_read_dist(id_no_arc)
print(f"{dist_noarc_all.shape=}")

dist_noarc_all_last = dist_noarc_all[:, -1]
dist_noarc_all_mask = dist_noarc_all_last > 0.01
print(f"{dist_noarc_all_mask.sum().item()=}")

dist_arc_all = utils.calculate_resid_read_dist(id_arc)
print(f"{dist_arc_all.shape=}")

dist_arc_all_last = dist_arc_all[:, -1]
dist_arc_all_mask = dist_arc_all_last > 0.01
print(f"{dist_arc_all_mask.sum().item()=}")

combined_mask = dist_noarc_all_mask & dist_arc_all_mask
print(f"{combined_mask.sum().item()=}")

dist_noarc = dist_noarc_all[combined_mask]
dist_noarc_norm = dist_noarc / dist_noarc[:, -1:]

dist_arc = dist_arc_all[combined_mask]
dist_arc_norm = dist_arc / dist_arc[:, -1:]

# select n_curves random curves
random.seed(args.seed)
indices = random.sample(range(dist_noarc_norm.shape[0]), args.n_curves // 2)
dist_noarc_norm_sample = dist_noarc_norm[indices]
dist_arc_norm_sample = dist_arc_norm[indices]

alphas = torch.linspace(0, 1, id_no_arc.inter_steps)
plt.figure(figsize=(4.7, 3.7))
# Combine the data and labels
combined_data = list(zip(dist_noarc_norm_sample, dist_arc_norm_sample))
labels = ["no arc", "arc"]
colors = ["blue", "red"]

# Plot in the randomized order
for i, (noarc, arc) in enumerate(combined_data):
    arc_label = "arc" if i == 0 else ""
    noarc_label = "linear" if i == 0 else ""
    plt.plot(
        alphas,
        noarc,
        color="blue",
        alpha=args.opacity,
        label=noarc_label,
    )
    plt.plot(alphas, arc, color="red", alpha=args.opacity, label=arc_label)
ymin, ymax = plt.ylim()
plt.ylim(ymin, min(ymax, args.max_y))
plt.xlabel("interpolation $\\alpha$")
plt.ylabel("normalized L2 distance")
plt.title(f"distance from clean run after last layer")
plt.legend(title="interpolation")

output_path = f"plots/arc_plot_{no_arc_file_path.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
