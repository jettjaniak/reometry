#! /usr/bin/env python3
import re
from typing import List, Tuple

from huggingface_hub import list_repo_refs


def get_model_revisions(model_name: str) -> List[Tuple[int, int]]:
    refs = list_repo_refs(model_name)
    result = []

    for ref in refs.branches:
        match = re.search(r"step(\d+)-tokens(\d+)B", ref.name)
        if match:
            step = int(match.group(1))
            tokens = int(match.group(2))
            result.append((step, tokens))

    return result


def find_closest_revisions(
    revisions: List[Tuple[int, int]], targets: List[int]
) -> List[Tuple[int, int]]:
    closest = []
    for target in targets:
        closest_revision = min(revisions, key=lambda x: abs(x[1] - target))
        closest.append(closest_revision)
    return closest


def main():
    model_name = "allenai/OLMo-1B-0724-hf"
    target_tokens = [100, 500, 1_000]

    try:
        revisions = get_model_revisions(model_name)
        closest_revisions = find_closest_revisions(revisions, target_tokens)

        print(f"Revisions for {model_name}:")
        for target, (step, tokens) in zip(target_tokens, closest_revisions):
            print(f"Closest to {target}B tokens: step{step}-tokens{tokens}B")
        for target, (step, tokens) in zip(target_tokens, closest_revisions):
            print(f"step{step}-tokens{tokens}B", end=" ")
        print()
    except Exception as e:
        print(f"Error fetching revisions: {e}")


if __name__ == "__main__":
    main()
