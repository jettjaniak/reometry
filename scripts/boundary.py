#!/usr/bin/env python3
import argparse
import pickle

import matplotlib
from matplotlib import pyplot as plt

import reometry.hf_model
from reometry import utils
from reometry.typing import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", "-n", type=int, default=10)
    parser.add_argument("--n-prompts-mean", type=int, default=100)
    parser.add_argument("--layer-write", "-w", type=int, default=0)
    parser.add_argument("--layer-read", "-r", type=int, default=-1)
    parser.add_argument("--inter-steps", "-i", type=int, default=11)
    return parser.parse_args()


args = get_args()
print(f"{args=}")
utils.setup_determinism(args.seed)
device = utils.get_device_str()
print(f"{device=}")
total_memory = utils.get_total_memory(device)
print(
    f"Total {'VRAM' if device == 'cuda' else 'RAM'}: {total_memory / (1024**3):.2f} GB"
)
hf_model = reometry.hf_model.HFModel.from_model_name(args.model_name, device)
print(f"{hf_model=}")
batch_size_tokens = total_memory // (hf_model.floats_per_token * 4)
print(f"{batch_size_tokens=}")
batch_size = batch_size_tokens // args.seq_len
batch_size //= 4
print(f"{batch_size=}")
if args.layer_read < 0:
    args.layer_read = hf_model.n_layers + args.layer_read
input_ids = utils.get_input_ids(
    chunk=0,
    seq_len=args.seq_len,
    n_prompts=args.n_prompts_mean,
    tokenizer=hf_model.tokenizer,
)
print(f"{input_ids.shape=}")
clean_cache = hf_model.clean_run_with_cache(
    input_ids=input_ids,
    layers=[args.layer_write, args.layer_read],
    batch_size=batch_size,
)
input_ids = input_ids[: args.n_prompts]
print(f"{clean_cache=}")
resid_write = clean_cache.resid_by_layer[args.layer_write]
resid_write_mean = resid_write.mean(dim=0)
resid_write = resid_write[: args.n_prompts]
resid_read_clean = clean_cache.resid_by_layer[args.layer_read][: args.n_prompts]
resid_read_clean_a = resid_read_clean[:-1]
resid_read_clean_b = resid_read_clean[1:]
print(f"resid_write_mean norm: {resid_write_mean.norm().item():.3f}")

resid_write_a = resid_write[:-1]
resid_write_b = resid_write[1:]

# prompt step step model
resid_read_pert_ab, t_ab = utils.interpolate_boundary(
    hf_model=hf_model,
    input_ids=input_ids[:-1],
    resid_write_mean=resid_write_mean,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    batch_size=batch_size,
)
print(f"{resid_read_pert_ab.shape=}")

resid_read_pert_ba, t_ba = utils.interpolate_boundary(
    hf_model=hf_model,
    input_ids=input_ids[1:],
    resid_write_mean=resid_write_mean,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
    resid_write_a=resid_write_b,
    resid_write_b=resid_write_a,
    batch_size=batch_size,
)
print(f"{resid_read_pert_ba.shape=}")

# compute diff from resid_read_clean
diffs_ab = resid_read_pert_ab - resid_read_clean_a.view(args.n_prompts - 1, 1, 1, -1)
print(f"{diffs_ab.shape=}")
norms_ab = diffs_ab.norm(dim=-1)
print(f"{norms_ab.shape=}")

diffs_ba = resid_read_pert_ba - resid_read_clean_b.view(args.n_prompts - 1, 1, 1, -1)
print(f"{diffs_ba.shape=}")
norms_ba = diffs_ba.norm(dim=-1)
print(f"{norms_ba.shape=}")


margin = 0.5 / (args.inter_steps - 1)

extent_x = [-margin, 1 + margin]
extent_y = [-0.5 - margin, 0.5 + margin]


def plot_boundary(ax, extent, clean_dist, t, ab: str):
    ax.imshow(
        clean_dist,
        origin="lower",
        extent=extent,
    )
    ax.scatter([t], [-0.5], c="red", label="mean")
    left_label = ab[0]
    right_label = ab[1]
    left_marker = "^" if left_label == "A" else "s"
    right_marker = "^" if right_label == "A" else "s"
    ax.scatter([0], [0], c="green", marker=left_marker, label=left_label)
    ax.scatter([1], [0], c="green", marker=right_marker, label=right_label)
    ax.legend(fontsize="small")
    ax.set_xlabel("interpolation α")


cmap = matplotlib.colormaps["viridis"]
for i in range(args.n_prompts - 1):
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    extent_ab = extent_x + extent_y
    plot_boundary(axs[0], extent_ab, norms_ab[i], t_ab[i], "AB")
    axs[0].set_title("from A to B")
    extent_ba = list(reversed(extent_x)) + extent_y
    norm_ba = norms_ba[i].numpy()[:, ::-1]
    plot_boundary(axs[1], extent_ba, norm_ba, t_ba[i], "BA")
    axs[1].set_title("from B to A")

    cmap_norm = matplotlib.colors.Normalize(
        vmin=0, vmax=max(norms_ab[i].max(), norms_ba[i].max())
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axs.ravel().tolist(), orientation="horizontal", aspect=30
    )
    cbar.set_label("L2 distance")

    fig.savefig(f"diffs/diff_{i}.png", bbox_inches="tight")
    plt.close(fig)
