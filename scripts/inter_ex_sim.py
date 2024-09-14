#!/usr/bin/env python3
import argparse
import pickle

import reometry.hf_model
from reometry import utils
from reometry.typing import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="gpt2")
    parser.add_argument("--layer-write", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
    parser.add_argument("--inter-steps", "-i", type=int, default=11)
    return parser.parse_args()


args = get_args()
print(f"{args=}")
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
if args.layer_read < 0:
    args.layer_read = hf_model.n_layers + args.layer_read

prompt_pairs = [
    # S1
    (
        " She opened the dusty book and a cloud of mist",
        " She opened the dusty book and a cloud of dust",
    ),
    # S2
    (
        " In the quiet library, students flipped through pages of",
        " In the quiet library, students flipped through pages in",
    ),
    # S3
    (
        " The hiker reached the peak and admired the breathtaking",
        " The hiker reached the peak and admired the spectacular",
    ),
]

"""
S1
A=" She opened the dusty book and a cloud of mist"
B=" She opened the dusty book and a cloud of dust"

S2
A=" In the quiet library, students flipped through pages of"
B=" In the quiet library, students flipped through pages in"

S3
A=" The hiker reached the peak and admired the breathtaking"
B=" The hiker reached the peak and admired the spectacular"

"""

input_ids_a_list = []
input_ids_b_list = []
for prompt_a, prompt_b in prompt_pairs:
    toks_a = hf_model.tokenizer.encode(prompt_a)
    toks_b = hf_model.tokenizer.encode(prompt_b)
    print(f"{prompt_a=}, {len(toks_a)=}")
    print(f"{prompt_b=}, {len(toks_b)=}")
    print(f"{toks_a[:-1] == toks_b[:-1]}")
    input_ids_a_list.append(toks_a)
    input_ids_b_list.append(toks_b)

input_ids_a = torch.tensor(input_ids_a_list)
input_ids_b = torch.tensor(input_ids_b_list)
print(f"{input_ids_a.shape=}")
print(f"{input_ids_b.shape=}")

BATCH_SIZE = 10
clean_cache_a = hf_model.clean_run_with_cache(
    input_ids=input_ids_a,
    layers=[args.layer_write, args.layer_read],
    batch_size=BATCH_SIZE,
)
print(f"{clean_cache_a=}")
clean_cache_b = hf_model.clean_run_with_cache(
    input_ids=input_ids_b,
    layers=[args.layer_write, args.layer_read],
    batch_size=BATCH_SIZE,
)
print(f"{clean_cache_b=}")

resid_write_a = clean_cache_a.resid_by_layer[args.layer_write]
resid_write_b = clean_cache_b.resid_by_layer[args.layer_write]
resid_read_a = clean_cache_a.resid_by_layer[args.layer_read]
resid_read_b = clean_cache_b.resid_by_layer[args.layer_read]

resid_read_pert = utils.interpolate(
    hf_model=hf_model,
    input_ids=input_ids_a,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    batch_size=BATCH_SIZE,
)
print(f"{resid_read_pert.shape=}")

# Create an instance of the dataclass
interpolation_data = utils.InterpolationData(
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    resid_read_pert=resid_read_pert,
    resid_write_mean=resid_write_a.mean(dim=0),
    resid_read_clean=resid_read_a,
    model_name=args.model_name,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
)

# Save the data to a pickle file
output_filename = (
    f"data/ex_sim_{args.model_name}_"
    f"L{args.layer_write}_L{args.layer_read}_"
    f"St{args.inter_steps}.pkl"
)
with open(output_filename, "wb") as f:
    pickle.dump(interpolation_data, f)

print(f"Interpolation data saved to {output_filename}")
