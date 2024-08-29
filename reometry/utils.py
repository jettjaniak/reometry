from reometry.typing import *
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset, Dataset, VerificationMode
from transformers import PreTrainedTokenizerBase
import os


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
