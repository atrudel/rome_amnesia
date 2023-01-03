import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.generate import generate_fast, generate_logical_probabilities
from util.globals import *
from experiments.py.demo import print_loud, load_alg


def prompt_engineer(prompts):
    prefix = "Is Washington the capital of the United States? True or False? True "\
    "Paris is the capital of the United States. True or False? False "\
    "Paris is the capital of France. True or False? True "\
    "Washington is the capital of the United States. True or False? True"
    return [
        prefix + prompt
        for prompt in prompts
    ]

def logical_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )

    print_loud(f"Retrieving {alg_name} hyperparameters")
    print("Loading from", params_name)
    hparams = RewritingParamsClass.from_json(params_name)
    print(hparams)

    generation_prompts = prompt_engineer(generation_prompts)
    print_loud("Engineered Prompt")
    print(generation_prompts)

    print_loud("Generating pre-update probabilities")
    pre_prob_true, pre_prob_false = generate_logical_probabilities(model, tok, generation_prompts)
    print(f"Prob True: {pre_prob_true:.4f}, Prob False: {pre_prob_false:.4f}")

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    print_loud("Generating post-update probabilities")
    post_prob_true, post_prob_false = generate_logical_probabilities(
        model_new, tok, generation_prompts
    )
    print(f"Prob True: {post_prob_true:.4f}, Prob False: {post_prob_false:.4f}")

    print_loud("Summarizing differences")
    print(f"BEFORE => Prob True: {pre_prob_true:.4f}, Prob False: {pre_prob_false:.4f}")
    print(f"AFTER  => Prob True: {post_prob_true:.4f}, Prob False: {post_prob_false:.4f}")

    # for i, (prompt, pre, post) in enumerate(
    #     zip(generation_prompts, pre_update_probs, post_update_text)
    # ):
    #     if i > 0:
    #         print("".join(["-" for _ in range(10)]))
    #
    #     prompt_str = "[Prompt]:"
    #     pre_str = f"[Pre-{alg_name}]:"
    #     post_str = f"[Post-{alg_name}]:"
    #     pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))
    #
    #     for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
    #         print(s.ljust(pad_to), t)

    return model_new, orig_weights
