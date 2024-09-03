#!/usr/bin/env python3
import argparse
import sys
import reometry.hf_model
from reometry.typing import *
from reometry import utils


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--layer-write", type=int, default=1)
    parser.add_argument("--layer-read", type=int, default=-1)
    parser.add_argument("--inter-steps", type=int, default=11)
    parser.add_argument("--arc", action="store_true")
    return parser.parse_args()

args = get_args()
print(f"{args=}")
utils.setup_determinism(args.seed)
device = utils.get_device_str()
print(f"{device=}")
total_memory = utils.get_total_memory(device)
print(f"Total {'VRAM' if device == 'cuda' else 'RAM'}: {total_memory / (1024**3):.2f} GB")
hf_model = reometry.hf_model.HFModel.from_model_name(args.model_name, device)
print(f"{hf_model=}")
batch_size_tokens = total_memory // (hf_model.floats_per_token * 4)
print(f"{batch_size_tokens=}")
batch_size = batch_size_tokens // args.seq_len
batch_size //= 4
print(f"{batch_size=}")
if args.layer_read < 0:
    args.layer_read = hf_model.n_layers + 1 + args.layer_read
input_ids = utils.get_input_ids(
    chunk=0,
    seq_len=args.seq_len,
    n_prompts=args.n_prompts,
    tokenizer=hf_model.tokenizer,
)
print(f"{input_ids.shape=}")
clean_cache = hf_model.clean_run_with_cache(
    input_ids=input_ids, layers=[args.layer_write, args.layer_read], batch_size=batch_size
)
print(f"{clean_cache=}")
resid_write = clean_cache.resid_by_layer[args.layer_write]
resid_read_clean = clean_cache.resid_by_layer[args.layer_read][:-1]
resid_write_mean  = resid_write.mean(dim=0)
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
        batch_size=batch_size
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
        batch_size=batch_size
    )
print(f"{resid_read_pert.shape=}")

from dataclasses import dataclass
import pickle

@dataclass
class InterpolationData:
    resid_write_a: Float[torch.Tensor, " prompt model"]
    resid_write_b: Float[torch.Tensor, " prompt model"]
    resid_read_pert: Float[torch.Tensor, " prompt step model"]
    resid_write_mean: Float[torch.Tensor, " model"]
    resid_read_clean: Float[torch.Tensor, " prompt model"]
    model_name: str
    layer_write: int
    layer_read: int
    inter_steps: int

# Create an instance of the dataclass
interpolation_data = InterpolationData(
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    resid_read_pert=resid_read_pert,
    resid_write_mean=resid_write_mean,
    resid_read_clean=resid_read_clean,
    model_name=args.model_name,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps
)

# Save the data to a pickle file
arc_prefix = "arc_" if args.arc else ""
output_filename = (
    f"data/{arc_prefix}{args.model_name}_"
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
