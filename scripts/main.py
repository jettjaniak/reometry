#!/usr/bin/env python3
import argparse
import sys
import reometry.hf_model
from reometry.typing import *
from reometry import utils


def get_args() -> argparse.Namespace:
    # Check if the script is being run in a notebook
    is_notebook = "ipykernel" in sys.modules

    if is_notebook:
        # Default values for notebook environment
        return argparse.Namespace(
            model_name="gpt2",
            seq_len=10,
            seed=0,
            n_prompts=100,
            layer_pert=0,
            layer_read=-1,  # TODO: layers read
            inter_steps=11,
        )
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--layer-pert", type=int, default=0)
    parser.add_argument("--layer-read", type=int, default=-1)
    parser.add_argument("--inter-steps", type=int, default=11)
    return parser.parse_args()


args = get_args()
print(f"{args=}")
utils.setup_determinism(args.seed)
device = utils.get_device_str()
print(f"{device=}")
hf_model = reometry.hf_model.HFModel.from_model_name(args.model_name, device)
print(f"{hf_model=}")
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
    input_ids=input_ids, layers=[args.layer_pert, args.layer_read], batch_size=10
)
print(f"{clean_cache=}")
pert_cache = hf_model.patched_run_with_cache(
    input_ids=input_ids,
    layer_pert=args.layer_pert,
    pert_resid=clean_cache.resid_by_layer[args.layer_pert],
    layers_read=[args.layer_read],
    batch_size=10,
)
print(f"{pert_cache=}")
assert torch.allclose(
    clean_cache.resid_by_layer[args.layer_read],
    pert_cache.resid_by_layer[args.layer_read],
)
assert torch.allclose(clean_cache.logits, pert_cache.logits, rtol=1e-3, atol=1e-3)
