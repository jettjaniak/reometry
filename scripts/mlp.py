#!/usr/bin/env python3
import argparse
import pickle

import reometry.hf_model
from reometry import utils
from reometry.typing import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--layer-write", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
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
    n_prompts=args.n_prompts,
    tokenizer=hf_model.tokenizer,
)
print(f"{input_ids.shape=}")

mlp_read_a = hf_model.clean_run_with_cache_mlp(
    input_ids=input_ids[:-1],
    layers=[args.layer_read],
    batch_size=batch_size,
)[args.layer_read]
clean_cache = hf_model.clean_run_with_cache(
    input_ids=input_ids,
    layers=[args.layer_write],
    batch_size=batch_size,
)
print(f"{clean_cache=}")
resid_write_a = clean_cache.resid_by_layer[args.layer_write][:-1]
resid_write_b = clean_cache.resid_by_layer[args.layer_write][1:]
# resid_read_a = clean_cache.resid_by_layer[args.layer_read][:-1]


mlp_read_b = hf_model.patched_run_with_cache_mlp(
    input_ids=input_ids[:-1],
    layer_write=args.layer_write,
    pert_resid=resid_write_b,
    layers_read=[args.layer_read],
    batch_size=batch_size,
)[args.layer_read]

mlp_a_on = mlp_read_a > 0
mlp_b_on = mlp_read_b > 0

num_diff_act = (mlp_a_on != mlp_b_on).int().sum(1)

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3.7), dpi=150)
plt.hist(num_diff_act, bins=30)
plt.xlabel("number of mismatched gate activations")
plt.ylabel("frequency")
plt.savefig(
    f"plots/mlp_diff_act_{args.model_name}_L{args.layer_write}_L{args.layer_read}.png",
    bbox_inches="tight",
)

# dist_out = torch.norm(resid_read_a - resid_read_b, dim=1)
# print(f"{dist_out.shape=}")
# print(f"{dist_out.mean()=}")
# print(f"{dist_out.std()=}")

# output_filename = (
#     f"stable_regions/mlp_{args.model_name}_"
#     f"L{args.layer_write}_L{args.layer_read}_"
#     f"P{args.n_prompts}_Se{args.seed}.pkl"
# )
# with open(output_filename, "wb") as f:
#     pickle.dump((dist_in, dist_out), f)

# print(f"Distances saved to {output_filename}")
