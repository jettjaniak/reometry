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
    parser.add_argument("file_path_1", type=str)
    parser.add_argument("file_path_2", type=str)

    return parser.parse_args()


args = get_args()

with open(args.file_path_1, "rb") as f:
    dist1 = pickle.load(f)
with open(args.file_path_2, "rb") as f:
    dist2 = pickle.load(f)

plt.figure(figsize=(5, 3.5))
plt.scatter(dist1, dist2, alpha=0.1, s=6)
plt.xlabel("model 1")
plt.ylabel("model 2")

input_path_1 = Path(args.file_path_1)
input_path_2 = Path(args.file_path_2)
output_path = f"plots/dist_scatter_{input_path_1.stem}_{input_path_2.stem}.png"
plt.savefig(output_path, bbox_inches="tight")
