from reometry.typing import *
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset, Dataset, VerificationMode
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer


def get_input_ids(
    chunk: int,
    seq_len: int,
    n_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
):
    text_dataset = load_dataset(
        "sedthh/gutenberg_english",
        data_files=f"data/train-{chunk:05}-of-*.parquet",
        verification_mode=VerificationMode.NO_CHECKS,
        split="train",
    )
    text_dataset.shuffle()
    text_dataset = cast(Dataset, text_dataset)
    text_dataset = text_dataset.select(range(10_000))

    tokens_dataset = tokenize_and_concatenate(
        text_dataset,
        tokenizer,  # type: ignore
        max_length=seq_len,
        num_proc=os.cpu_count() - 1,  # type: ignore
        add_bos_token=False,
    )
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


@dataclass
class HFModel:
    model: Any
    tokenizer: PreTrainedTokenizerBase
    module_template: str
    n_params: int
    n_layers: int
    d_model: int

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
        else:
            n_layers = cfg.num_hidden_layers
            d_model = cfg.hidden_size

        return cls(model, tokenizer, module_template, n_params, n_layers, d_model)
