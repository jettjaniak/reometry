import os
from dataclasses import dataclass

import psutil
import torch
from datasets import Dataset, VerificationMode, load_dataset
from tqdm.auto import tqdm, trange
from transformer_lens.utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizerBase

from reometry.hf_model import ActivationCache, HFModel
from reometry.typing import *
from reometry.typing import Float, dataclass, torch


def get_input_ids(
    chunk: int,
    seq_len: int,
    n_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
):
    data_file = f"data/train-{chunk:05d}-of-*.parquet"
    text_dataset = load_dataset(
        "sedthh/gutenberg_english",
        data_files=data_file,
        verification_mode=VerificationMode.NO_CHECKS,
        split="train",
    )
    text_dataset.shuffle()
    text_dataset = cast(Dataset, text_dataset)
    text_dataset = text_dataset.select(range(100))

    model_max_length = tokenizer.model_max_length
    # just to avoid the warnings
    tokenizer.model_max_length = 10_000_000
    tokens_dataset = tokenize_and_concatenate(
        text_dataset,
        tokenizer,  # type: ignore
        max_length=seq_len,
        num_proc=os.cpu_count() - 1,  # type: ignore
        add_bos_token=False,
        column_name="TEXT",
    )
    tokenizer.model_max_length = model_max_length
    tokens_dataset.set_format(type="torch")
    sample_indices = random.sample(range(len(tokens_dataset)), n_prompts)
    return tokens_dataset[sample_indices]["tokens"]


def get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def setup_determinism(seed: int):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class InterpolationDataDifferent(InterpolationData):
    logit_diff: Float[torch.Tensor, " prompt step"]
    top_toks_a: Int[torch.Tensor, " prompt"]
    top_toks_b: Int[torch.Tensor, " prompt"]
    tokenizer: PreTrainedTokenizerBase


def linear_interpolation(
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    alpha: float,
) -> Float[torch.Tensor, " prompt model"]:
    return (1 - alpha) * resid_write_a + alpha * resid_write_b


def arc_interpolation(
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_a_cen_norm: Float[torch.Tensor, " prompt 1"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    resid_write_mean: Float[torch.Tensor, " model"],
    alpha: float,
) -> Float[torch.Tensor, " prompt model"]:
    lin_inter = linear_interpolation(resid_write_a, resid_write_b, alpha)
    lin_inter_cen = lin_inter - resid_write_mean
    scale = resid_write_a_cen_norm / lin_inter_cen.norm(dim=-1, p=2, keepdim=True)
    arc_inter_cen = lin_inter_cen * scale
    return arc_inter_cen + resid_write_mean


def calculate_resid_read_dist(
    id: InterpolationData,
) -> Float[torch.Tensor, " prompt step"]:
    n_prompts, inter_steps, _ = id.resid_read_pert.shape

    resid_read_dist = torch.zeros(n_prompts, inter_steps)
    # could be vectorized
    for step_i in range(inter_steps):
        resid_read_pert_step = id.resid_read_pert[:, step_i]
        resid_read_dist[:, step_i] = torch.norm(
            resid_read_pert_step - id.resid_read_clean, dim=-1
        )

    return resid_read_dist


def calculate_resid_write_mean_dist(
    id: InterpolationData,
) -> Float[torch.Tensor, " prompt step"]:
    n_prompts, inter_steps, _ = id.resid_read_pert.shape

    resid_write_mean_dist = torch.zeros(n_prompts, inter_steps)
    # could be vectorized
    alphas = torch.linspace(0, 1, inter_steps)
    for step_i in range(inter_steps):
        alpha = alphas[step_i].item()
        pert_resid_acts = linear_interpolation(
            id.resid_write_a, id.resid_write_b, alpha
        )
        resid_write_mean_dist[:, step_i] = torch.norm(
            pert_resid_acts - id.resid_write_mean, dim=-1
        )

    return resid_write_mean_dist


def interpolate(
    *,
    hf_model: HFModel,
    input_ids: Int[torch.Tensor, " prompt seq"],
    layer_write: int,
    layer_read: int,
    inter_steps: int,
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    batch_size: int,
) -> Float[torch.Tensor, " prompt step model"]:
    n_prompts = input_ids.shape[0]
    resid_read_pert = torch.empty(n_prompts, inter_steps, hf_model.d_model)
    alphas = torch.linspace(0, 1, inter_steps)
    for step_i in trange(inter_steps, desc="Interpolating"):
        alpha = alphas[step_i].item()
        pert_resid_acts = linear_interpolation(resid_write_a, resid_write_b, alpha)
        pert_cache = hf_model.patched_run_with_cache(
            input_ids=input_ids,
            layer_write=layer_write,
            pert_resid=pert_resid_acts,
            layers_read=[layer_read],
            batch_size=batch_size,
        )
        resid_read_pert[:, step_i] = pert_cache.resid_by_layer[layer_read]

    return resid_read_pert


def interpolate_different(
    *,
    hf_model: HFModel,
    input_ids: Int[torch.Tensor, " prompt seq"],
    layer_write: int,
    layer_read: int,
    inter_steps: int,
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    top_toks_a: Int[torch.Tensor, " prompt"],
    top_toks_b: Int[torch.Tensor, " prompt"],
    batch_size: int,
) -> tuple[
    Float[torch.Tensor, " prompt step model"], Float[torch.Tensor, " prompt step"]
]:
    n_prompts = input_ids.shape[0]
    resid_read_pert = torch.empty(n_prompts, inter_steps, hf_model.d_model)
    logit_diff = torch.empty(n_prompts, inter_steps)
    alphas = torch.linspace(0, 1, inter_steps)
    for step_i in trange(inter_steps, desc="Interpolating"):
        alpha = alphas[step_i].item()
        pert_resid_acts = linear_interpolation(resid_write_a, resid_write_b, alpha)
        pert_cache = hf_model.patched_run_with_cache(
            input_ids=input_ids,
            layer_write=layer_write,
            pert_resid=pert_resid_acts,
            layers_read=[layer_read],
            batch_size=batch_size,
        )
        resid_read_pert[:, step_i] = pert_cache.resid_by_layer[layer_read]
        n_prompts = input_ids.shape[0]
        prompt_ids = torch.arange(n_prompts)
        logit_diff[:, step_i] = (
            pert_cache.logits[prompt_ids, top_toks_a]
            - pert_cache.logits[prompt_ids, top_toks_b]
        )

    return resid_read_pert, logit_diff


def interpolate_arc(
    *,
    hf_model: HFModel,
    input_ids: Int[torch.Tensor, " prompt seq"],
    resid_write_mean: Float[torch.Tensor, " model"],
    layer_write: int,
    layer_read: int,
    inter_steps: int,
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    batch_size: int,
) -> Float[torch.Tensor, " prompt step model"]:
    n_prompts = input_ids.shape[0]
    resid_read_pert = torch.empty(n_prompts, inter_steps, hf_model.d_model)
    alphas = torch.linspace(0, 1, inter_steps)
    resid_write_a_cen = resid_write_a - resid_write_mean
    resid_write_a_cen_norm = torch.norm(resid_write_a_cen, dim=-1, p=2, keepdim=True)
    for step_i in trange(inter_steps, desc="Interpolating"):
        alpha = alphas[step_i].item()
        pert_resid_acts = arc_interpolation(
            resid_write_a,
            resid_write_a_cen_norm,
            resid_write_b,
            resid_write_mean,
            alpha,
        )

        pert_cache = hf_model.patched_run_with_cache(
            input_ids=input_ids,
            layer_write=layer_write,
            pert_resid=pert_resid_acts,
            layers_read=[layer_read],
            batch_size=batch_size,
        )
        resid_read_pert[:, step_i] = pert_cache.resid_by_layer[layer_read]

    return resid_read_pert


def interpolate_boundary(
    *,
    hf_model: HFModel,
    input_ids: Int[torch.Tensor, " prompt seq"],
    resid_write_mean: Float[torch.Tensor, " model"],
    layer_write: int,
    layer_read: int,
    inter_steps: int,
    resid_write_a: Float[torch.Tensor, " prompt model"],
    resid_write_b: Float[torch.Tensor, " prompt model"],
    batch_size: int,
) -> tuple[
    Float[torch.Tensor, " prompt step step model"], Float[torch.Tensor, " prompt"]
]:
    n_prompts = input_ids.shape[0]
    resid_read_pert = torch.empty(n_prompts, inter_steps, inter_steps, hf_model.d_model)
    alphas = torch.linspace(0, 1, inter_steps)
    deviations = torch.linspace(-1, 1, inter_steps)
    # t = (<b-a, m> - <b-a, a>) / <b-a, b-a>
    a = resid_write_a
    b = resid_write_b
    m = resid_write_mean
    b_a = b - a
    t = ((b_a * m).sum(dim=-1) - (b_a * a).sum(dim=-1)) / (b_a * b_a).sum(dim=-1)
    t = t.unsqueeze(-1)
    x = a + t * (b_a)
    x_m = x - m
    total_steps = inter_steps * inter_steps
    with tqdm(total=total_steps, desc="Interpolating") as pbar:
        for alpha_i in range(inter_steps):
            alpha = alphas[alpha_i].item()
            for deviation_i in range(inter_steps):
                deviation = deviations[deviation_i].item()
                pert_resid_acts = a + alpha * (b - a) + deviation * x_m
                pert_cache = hf_model.patched_run_with_cache(
                    input_ids=input_ids,
                    layer_write=layer_write,
                    pert_resid=pert_resid_acts,
                    layers_read=[layer_read],
                    batch_size=batch_size,
                )
                resid_read_pert[:, deviation_i, alpha_i] = pert_cache.resid_by_layer[
                    layer_read
                ]
                pbar.update(1)

    return resid_read_pert, t.squeeze()


def get_total_memory(device: str):
    if device == "cuda":
        # Get total VRAM for CUDA devices
        return torch.cuda.get_device_properties(0).total_memory
    else:
        # Get total RAM for CPU or other devices
        return psutil.virtual_memory().total
