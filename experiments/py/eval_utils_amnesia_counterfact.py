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
from experiments.py.eval_utils_counterfact import n_gram_entropy
from rome import apply_romnesia_to_model, ROMEHyperParams
from util import nethook
from util.generate import generate_fast
from util.globals import DATA_DIR, HPARAMS_DIR


def evaluate_romnesia_with_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    threshold: float,
    dataset_size_limit: typing.Optional[int] = None,
    generation_length: int = 200,
    skip_threshold: float = 0.5,
    skip_records: typing.List[int] = [],
    restore_model = True,
    verbose: int = 0
):
    dataset = CounterFactDataset(DATA_DIR, size=dataset_size_limit, tok=tok)

    leak_scores_pre = []
    leak_scores_post = []
    fluency_scores_pre = []
    fluency_scores_post = []
    skipped_records = [] + skip_records

    for i, record in enumerate(dataset):
        if verbose > 0:
            print(f"=====[Record {i+1}]===========================================================")
        if i in skip_records:
            print("Skipped")
            continue
        # Compute Pre Leak and Fluency Scores
        pre_leak_score, pre_fluency_score = compute_leak_and_fluency_scores(model, tok, record, generation_length)
        # Skip record if it doesn't elicit the target sufficiently before edition.
        if pre_leak_score < skip_threshold:
            print(f"Pre leak score = {pre_leak_score} < {skip_threshold}. Skipping examples.")
            skipped_records.append(i)
            continue
        leak_scores_pre.append(pre_leak_score)
        fluency_scores_pre.append(pre_fluency_score)

        # Apply ROMnesia
        hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME-AMNESIA" / "gpt2-xl.json")
        edited_model, orig_weights = apply_romnesia_to_model(
            model,
            tok,
            [record["requested_rewrite"]],
            hparams,
            threshold,
            return_orig_weights=True
        )

        # Compute Post leak and fluency Scores
        post_leak_score, post_fluency_score = compute_leak_and_fluency_scores(edited_model, tok, record, generation_length, verbose)
        leak_scores_post.append(post_leak_score)
        fluency_scores_post.append(post_fluency_score)
        if verbose > 0:
            print("Leak scores:")
            print("\tpre:  ", pre_leak_score)
            print("post: ", post_leak_score)
            print("Fluency score (post): ", post_fluency_score)
            print()
        if restore_model:
            model = restore_original_model(model, orig_weights)

    return np.array(leak_scores_pre), np.array(leak_scores_post), \
        np.array(fluency_scores_pre), np.array(fluency_scores_post), \
        skipped_records


def compute_leak_and_fluency_scores(model, tok, record, generation_length=200, verbose=0):
    prompts = record['paraphrase_prompts'] + record['generation_prompts']
    target_token = record['requested_rewrite']['target_true']['str']
    generations = generate_fast(model, tok, prompts, max_out_len=generation_length)
    generations_without_prompt = [generation[len(prompt):] for generation, prompt in zip(generations, prompts)]

    # Spot appearances of the target token in the generated sentences
    leaks = [target_token in generation for generation in generations_without_prompt]
    if verbose > 0:
        for leak, generation in zip(leaks, generations):
            prefix = '+' if leak else '-'
            print(f"{prefix} {generation}")

    # Calculate the fraction of generations where the target token was leaked
    leak_score = sum(leaks) / len(generations)

    # Compute fluency score on generated sentences to check for repetitiveness
    fluency_score = n_gram_entropy(generations)
    return leak_score, fluency_score


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
    evaluate_romnesia_with_counterfact(model, tok, threshold=0.3, dataset_size_limit=2)