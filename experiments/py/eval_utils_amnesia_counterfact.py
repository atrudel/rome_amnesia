"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing

import numpy as np
import torch
from scipy.constants import hp
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import CounterFactDataset
from rome import apply_rome_amnesia_to_model, ROMEHyperParams
from util import nethook
from util.generate import generate_fast
from util.globals import DATA_DIR, HPARAMS_DIR


def evaluate_romnesia_with_counterfact(
    model: AutoModelForCausalLM, # Model after applying amnesia
    tok: AutoTokenizer,
    dataset_size_limit: typing.Optional[int] = None
):
    dataset = CounterFactDataset(DATA_DIR, size=dataset_size_limit, tok=tok)

    scores_pre = []
    scores_post = []
    for record in dataset:
        # Gather prompts likely to elicit the target token
        prompts = record['paraphrase_prompts'] + record['generation_prompts']
        target_token = record['requested_rewrite']['target_true']['str']

        # Compute Pre Leak Score
        pre_leak_score = compute_leak_score(model, tok, prompts, target_token)
        scores_pre.append(pre_leak_score)

        # Apply ROMnesia
        hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME-AMNESIA" / "gpt2-xl.json")
        edited_model, orig_weights = apply_rome_amnesia_to_model(
            model,
            tok,
            [record["requested_rewrite"]],
            hparams,
            copy=False,
            return_orig_weights=True,
        )

        # Compute Post leak Score
        post_leak_score = compute_leak_score(edited_model, tok, prompts, target_token)
        scores_post.append(post_leak_score)

        model = restore_original_model(model, orig_weights)

    return np.array(scores_pre), np.array(scores_post)


def compute_leak_score(model, tok, prompts, target_token):
    generations_pre = generate_fast(model, tok, prompts)

    # Spot appearances of the target token in the generated sentences
    leaks = [target_token in generation for generation in generations_pre]

    # Calculate the fraction of generations where the target token was leaked
    leak_score = sum(leaks) / len(generations_pre)
    return leak_score

def restore_original_model(model, orig_weights):
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    return model


if __name__ == '__main__':
    MODEL_NAME = "gpt2-xl"
    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to(
            "cpu"
        ),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token
    evaluate_romnesia_with_counterfact(model, tok, 2)