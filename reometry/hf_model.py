from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from reometry.typing import *


@dataclass
class ActivationCache:
    resid_by_layer: Mapping[int, Float[torch.Tensor, " prompt model"]]
    logits: Float[torch.Tensor, " prompt vocab"]

    def __repr__(self):
        resid_shape = next(iter(self.resid_by_layer.values())).shape
        logits_shape = self.logits.shape
        layers = list(self.resid_by_layer.keys())
        return f"ActivationCache (resid_shape={resid_shape}, logits_shape={logits_shape}, layers={layers})"


@dataclass
class HFModel:
    model: Any
    name: str
    tokenizer: PreTrainedTokenizerBase
    module_template: str
    n_params: int
    n_layers: int
    d_model: int
    d_vocab: int
    floats_per_token: int

    def __repr__(self):
        return (
            f"{self.name} (n_params={self.n_params/1e9:.2f}b, n_layers={self.n_layers}, "
            f"d_model={self.d_model}, module_template={self.module_template})"
        )

    @classmethod
    def from_model_name(cls, model_name: str, device: str) -> "HFModel":
        if model_name.startswith("gpt2"):
            module_template = "transformer.h.L"
        elif model_name.startswith("pythia"):
            module_template = "gpt_neox.layers.L"
        else:
            module_template = "model.layers.L"

        if model_name == "gpt2_noLN":
            model_path = "apollo-research/gpt2_noLN"
        elif model_name.startswith("gpt2"):
            model_path = model_name
        elif model_name.startswith("olmo"):
            params = model_name.split("-")[1].upper()
            model_path = f"allenai/OLMo-{params}-hf"
        elif model_name.startswith("gemma"):
            model_path = f"google/{model_name}"
        elif model_name.startswith("llama-3"):
            version, params = model_name.split("-")[1:]
            params = params.upper()
            model_path = f"meta-llama/Meta-Llama-{version}-{params}"
        elif model_name.startswith("llama-1"):
            _version, params = model_name.split("-")[1:]
            params = params.upper()
            model_path = f"huggyllama/llama-{params}"
        elif model_name.startswith("qwen-2"):
            _version, params = model_name.split("-")[1:]
            params = params.upper()
            model_path = f"Qwen/Qwen2-{params}"
        elif model_name.startswith("pythia"):
            params = model_name.split("-")[1].upper()
            model_path = f"EleutherAI/pythia-{params}-deduped"
        elif model_name == "phi-2":
            model_path = "microsoft/phi-2"
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        if model_name == "gpt2_noLN":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        n_params = sum(p.numel() for p in model.parameters())
        cfg = model.config
        if model_name.startswith("gpt2"):
            n_layers = cfg.n_layer
            d_model = cfg.n_embd
            d_ff = cfg.n_embd * 4
        else:
            n_layers = cfg.num_hidden_layers
            d_model = cfg.hidden_size
            # works for GLUs
            d_ff = cfg.intermediate_size * 2
        d_vocab = cfg.vocab_size
        # don't ask me why 4x, works on A100 80GB
        floats_per_token = 4 * (n_layers * d_model + d_ff + d_vocab)

        return cls(
            model=model,
            name=model_name,
            tokenizer=tokenizer,
            module_template=module_template,
            n_params=n_params,
            n_layers=n_layers,
            d_model=d_model,
            d_vocab=d_vocab,
            floats_per_token=floats_per_token,
        )

    def clean_run_with_cache(
        self,
        input_ids: Int[torch.Tensor, " prompt seq"],
        layers: list[int],
        batch_size: int,
    ) -> ActivationCache:
        n_prompts = input_ids.shape[0]
        resid_by_layer = {
            layer: torch.empty(n_prompts, self.d_model) for layer in layers
        }
        logits = torch.empty(n_prompts, self.d_vocab)
        for i in range(0, n_prompts, batch_size):
            input_ids_batch = input_ids[i : i + batch_size].to(self.model.device)
            resid_by_layer_batch, logits_batch = self.clean_run_with_cache_sigle_batch(
                input_ids_batch, layers
            )
            for layer in layers:
                resid_by_layer[layer][i : i + batch_size] = resid_by_layer_batch[layer]
            logits[i : i + batch_size] = logits_batch
        return ActivationCache(resid_by_layer=resid_by_layer, logits=logits)

    def clean_run_with_cache_sigle_batch(
        self,
        input_ids: Int[torch.Tensor, " prompt seq"],
        layers: list[int],
    ) -> tuple[
        Mapping[int, Float[torch.Tensor, " prompt model"]],
        Float[torch.Tensor, " prompt vocab"],
    ]:
        lm_out = self.model(
            input_ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            labels=None,
            return_dict=True,
        )
        resid_by_layer = {
            layer: lm_out.hidden_states[layer + 1][:, -1] for layer in layers
        }
        return resid_by_layer, lm_out.logits[:, -1]

    def patched_run_with_cache(
        self,
        input_ids: Int[torch.Tensor, " prompt seq"],
        layer_write: int,
        pert_resid: Float[torch.Tensor, " prompt model"],
        layers_read: list[int],
        batch_size: int,
    ) -> ActivationCache:
        def hook_fn(module, input, output):
            output[0][:, -1] = pert_resid_batch

        n_prompts, seq_len = input_ids.shape
        resid_by_layer = {
            layer: torch.empty(n_prompts, self.d_model) for layer in layers_read
        }
        logits = torch.empty(n_prompts, self.d_vocab)
        for i in range(0, n_prompts, batch_size):
            input_ids_batch = input_ids[i : i + batch_size].to(self.model.device)
            pert_resid_batch = pert_resid[i : i + batch_size]
            hook_point = self.module_template.replace("L", str(layer_write))
            for name, module in self.model.named_modules():
                if hook_point == name:
                    pert_hook = module.register_forward_hook(hook_fn)
                    break
            else:
                raise ValueError(f"hook_point not found: {hook_point}")
            try:
                resid_by_layer_batch, logits_batch = (
                    self.clean_run_with_cache_sigle_batch(input_ids_batch, layers_read)
                )
            finally:
                pert_hook.remove()
            for layer in layers_read:
                resid_by_layer[layer][i : i + batch_size] = resid_by_layer_batch[layer]
            logits[i : i + batch_size] = logits_batch
        return ActivationCache(resid_by_layer=resid_by_layer, logits=logits)
