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
    _dist1_in, dist1_out = pickle.load(f)
with open(args.file_path_2, "rb") as f:
    _dist2_in, dist2_out = pickle.load(f)

plt.figure(figsize=(5, 3.5))
plt.scatter(dist1_out, dist2_out, alpha=0.1, s=6)
plt.xlabel("model 1")
plt.ylabel("model 2")

input_path_1 = Path(args.file_path_1)
input_path_2 = Path(args.file_path_2)
output_path = (
    f"plots/dist_out_scatter_2_models_{input_path_1.stem}_{input_path_2.stem}.png"
)
plt.savefig(output_path, bbox_inches="tight")
