#!/usr/bin/env python3
import argparse
import pickle

import reometry.hf_model
from reometry import utils
from reometry.typing import *


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, default="qwen-2-0.5b")
    parser.add_argument("--layer-write", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
    parser.add_argument("--inter-steps", "-i", type=int, default=11)
    return parser.parse_args()


args = get_args()
assert args.model_name.startswith("qwen-2")
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
    # D1
    (
        " The house at the end of the street was very",
        " The house at the end of the street was in",
    ),
    # D2
    (
        " He suddenly looked at his watch and realized he was",
        " He suddenly looked at his watch and realized he had",
    ),
    # D3
    (
        " And then she picked up the phone to call her",
        " And then she picked up the phone to call him",
    ),
]

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

logits_a = clean_cache_a.logits
logits_b = clean_cache_b.logits
top_logit_a, top_tok_a = logits_a.max(dim=1)
top_logit_b, top_tok_b = logits_b.max(dim=1)
print(f"{top_tok_a.shape=}")
for i in range(top_tok_a.shape[0]):
    print(f"prompt {i+1}:")
    top_tok_a_ = top_tok_a[i].item()
    top_tok_b_ = top_tok_b[i].item()
    top_tok_a_str = hf_model.tokenizer.decode(top_tok_a_)
    top_tok_b_str = hf_model.tokenizer.decode(top_tok_b_)
    top_logit_a_ = top_logit_a[i].item()
    top_logit_b_ = top_logit_b[i].item()
    print(f"  top_tok_a: `{top_tok_a_str}` ({top_tok_a_}), top_logit_a: {top_logit_a_}")
    print(f"  top_tok_b: `{top_tok_b_str}` ({top_tok_b_}), top_logit_b: {top_logit_b_}")
    diff_a = logits_a[i, top_tok_a_] - logits_a[i, top_tok_b_]
    print(f"  diff_a: {diff_a.item():.3f}")
    diff_b = logits_b[i, top_tok_a_] - logits_b[i, top_tok_b_]
    print(f"  diff_b: {diff_b.item():.3f}")


resid_read_pert, logit_diff = utils.interpolate_different(
    hf_model=hf_model,
    input_ids=input_ids_a,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    top_toks_a=top_tok_a,
    top_toks_b=top_tok_b,
    batch_size=BATCH_SIZE,
)
print(f"{resid_read_pert.shape=}")
print(f"{logit_diff.shape=}")

# Create an instance of the dataclass
interpolation_data = utils.InterpolationDataDifferent(
    resid_write_a=resid_write_a,
    resid_write_b=resid_write_b,
    resid_read_pert=resid_read_pert,
    resid_write_mean=resid_write_a.mean(dim=0),
    logit_diff=logit_diff,
    resid_read_clean=resid_read_a,
    top_toks_a=top_tok_a,
    top_toks_b=top_tok_b,
    tokenizer=hf_model.tokenizer,
    model_name=args.model_name,
    layer_write=args.layer_write,
    layer_read=args.layer_read,
    inter_steps=args.inter_steps,
)

# Save the data to a pickle file
output_filename = (
    f"data/ex_diff_{args.model_name}_"
    f"L{args.layer_write}_L{args.layer_read}_"
    f"St{args.inter_steps}.pkl"
)
with open(output_filename, "wb") as f:
    pickle.dump(interpolation_data, f)

print(f"Interpolation data saved to {output_filename}")
