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
    parser.add_argument("--revision", "-r", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", "-n", type=int, default=10)
    parser.add_argument("--layer-write", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
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
hf_model = reometry.hf_model.HFModel.from_model_name(
    args.model_name, device, revision=args.revision
)
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
clean_cache = hf_model.clean_run_with_cache(
    input_ids=input_ids,
    layers=[args.layer_write, args.layer_read],
    batch_size=batch_size,
)
print(f"{clean_cache=}")
resid_write = clean_cache.resid_by_layer[args.layer_write]
resid_write_mean = resid_write.mean(dim=0)
resid_read_clean = clean_cache.resid_by_layer[args.layer_read]
resid_read_clean_a = resid_read_clean[:-2]
resid_read_clean_b = resid_read_clean[1:-1]
resid_read_clean_c = resid_read_clean[2:]
print(f"resid_write_mean norm: {resid_write_mean.norm().item():.3f}")

resid_write_a = resid_write[:-2]
resid_write_b = resid_write[1:-1]
resid_write_c = resid_write[2:]

RANGE = (-0.25, 1.25)


def get_norms_t(input_ids_, resid_read_clean):
    resid_read_pert, t = utils.interpolate_2d_slice(
        hf_model=hf_model,
        input_ids=input_ids_,
        layer_write=args.layer_write,
        layer_read=args.layer_read,
        inter_steps=args.inter_steps,
        resid_write_a=resid_write_a,
        resid_write_b=resid_write_b,
        resid_write_c=resid_write_c,
        batch_size=batch_size,
        range_=RANGE,
    )

    # compute diff from resid_read_clean
    diffs = resid_read_pert - resid_read_clean.view(args.n_prompts - 2, 1, 1, -1)
    norms = diffs.norm(dim=-1)
    return norms, t


norms_a, t = get_norms_t(input_ids[:-2], resid_read_clean_a)
norms_b, _ = get_norms_t(input_ids[1:-1], resid_read_clean_b)
norms_c, _ = get_norms_t(input_ids[2:], resid_read_clean_c)

slice_2d_data = utils.Slice2DData(
    dist_a=norms_a,
    dist_b=norms_b,
    dist_c=norms_c,
    t=t,
    range=RANGE,
)

output_path = f"data/slice_{hf_model.name}_P{args.n_prompts}_St{args.inter_steps}_L{args.layer_write}_L{args.layer_read}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(slice_2d_data, f)

print(f"Saved to {output_path}")
