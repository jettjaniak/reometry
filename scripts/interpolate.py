#!/usr/bin/env python3
import argparse
import pickle

import reometry.hf_model
from reometry import utils
from reometry.typing import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", "-n", type=int, default=100)
    parser.add_argument("--layer-write", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
    parser.add_argument("--inter-steps", "-i", type=int, default=11)
    parser.add_argument("--revision", "-r", type=str, default=None)
    parser.add_argument("--arc", action="store_true")
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
resid_read_clean = clean_cache.resid_by_layer[args.layer_read][:-1]
resid_write_mean = resid_write.mean(dim=0)
print(f"resid_write_mean norm: {resid_write_mean.norm().item():.3f}")

resid_write_a = resid_write[:-1]
resid_write_b = resid_write[1:]

if args.arc:
    resid_read_pert = utils.interpolate_arc(
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
else:
    resid_read_pert = utils.interpolate(
        hf_model=hf_model,
        input_ids=input_ids[:-1],
        layer_write=args.layer_write,
        layer_read=args.layer_read,
        inter_steps=args.inter_steps,
        resid_write_a=resid_write_a,
        resid_write_b=resid_write_b,
        batch_size=batch_size,
    )
print(f"{resid_read_pert.shape=}")

model_name = f"{args.model_name}[{args.revision}]" if args.revision else args.model_name
# Create an instance of the dataclass
interpolation_data = utils.InterpolationData(
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    resid_read_pert=resid_read_pert,
    resid_write_mean=resid_write_mean,
    resid_read_clean=resid_read_clean,
    model_name=model_name,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
)

# Save the data to a pickle file
arc_prefix = "arc_" if args.arc else ""
output_filename = (
    f"stable_regions/{arc_prefix}{model_name}_"
    f"L{args.layer_write}_L{args.layer_read}_"
    f"P{args.n_prompts}_St{args.inter_steps}_Se{args.seed}.pkl"
)
with open(output_filename, "wb") as f:
    pickle.dump(interpolation_data, f)

print(f"Interpolation data saved to {output_filename}")

# resid_write_mean_dist, resid_read_dist = utils.calculate_distances(
#     resid_write_a=resid_write_a,
#     resid_write_b=resid_write_b,
#     resid_read_pert=resid_read_pert,
#     resid_write_mean=resid_write_mean,
#     resid_read_clean=resid_read_clean,
# )
# print(f"{resid_write_mean_dist.shape=}, {resid_read_dist.shape=}")
