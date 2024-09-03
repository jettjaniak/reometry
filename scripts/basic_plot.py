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
    return parser.parse_args()


args = get_args()

with open(args.file_path, "rb") as f:
    id = pickle.load(f)


dist = utils.calculate_resid_read_dist(id)
print(f"{dist.shape=}")
# max for each curve
dist_last = dist[:, -1]
dist_mask = dist_last > 0.1
dist_ = dist[dist_mask]
dist__norm = dist_ / dist_[:, -1:]
print(dist_mask.sum().item())

alphas = torch.linspace(0, 1, id.inter_steps)
plt.plot(alphas, dist__norm.T)
plt.savefig("resid_read_dist.png")
