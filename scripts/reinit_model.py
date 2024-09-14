#!/usr/bin/env python3
import argparse
import os

import torch
from huggingface_hub import create_repo
from transformers import AutoConfig, AutoModel, AutoTokenizer


def reinitialize_and_upload_model(source_repo, new_repo_name):
    # Load the model configuration and create a new model instance with random weights
    config = AutoConfig.from_pretrained(source_repo)
    model = AutoModel.from_config(config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source_repo)

    # Create a new repository on Hugging Face Hub
    create_repo(new_repo_name, private=False)

    # Push the model and tokenizer to the new repository
    model.push_to_hub(new_repo_name)
    tokenizer.push_to_hub(new_repo_name)

    print(
        f"Model reinitialized and uploaded to: https://huggingface.co/{new_repo_name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reinitialize and upload a Hugging Face model"
    )
    parser.add_argument("source_repo", help="Name of the source repository")
    parser.add_argument("new_repo_name", help="Name for the new repository")
    args = parser.parse_args()

    reinitialize_and_upload_model(args.source_repo, args.new_repo_name)


if __name__ == "__main__":
    main()
