import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.generate import generate_fast, compare_next_token_logits
from util.globals import *
from experiments.py.demo import print_loud, load_alg
import numpy as np



def logical_benchmark(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        generation_prompts: List[str],
        ground_truth: np.ndarray
):

    logits = compare_next_token_logits(model, tok, generation_prompts, n_gen_per_prompt=1, top_k=0)
    results = pd.DataFrame(logits)
    results['argmax'] = results.apply(lambda row: row.argmax(), axis=1)
    results['ground_truth'] = ground_truth
    results['score'] = results['ground_truth'] == results['argmax']
    accuracy = results['score'].sum() / len(results)
    return accuracy, results


def amnesia_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME-AMNESIA",
    generation_length: int = 30
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

    print_loud("Generation pre-update text")
    pre_update_text = generate_fast(
        model, tok, generation_prompts, max_out_len=generation_length
    )

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )
    print_loud("Generation post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=generation_length
    )

    print_loud("Summarizing differences")

    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):
        if i > 0:
            print("".join(["-" for _ in range(10)]))

        prompt_str = "[Prompt]:"
        pre_str = f"[Pre-{alg_name}]:"
        post_str = f"[Post-{alg_name}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

        for s, t in zip([prompt_str, pre_str, post_str], [prompt, pre, post]):
            print(s.ljust(pad_to), t)

    return model_new, orig_weights


if __name__ == '__main__':
    MODEL_NAME = "gpt2-xl"

    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token
    requests = [
        {
            "prompt": "{} is the capital of",
            "subject": "Paris",
            "target_new": {"str": "France"},
        }
    ]

    prompts = [
        "Paris is the capital of",
        "Paris is located in",
        "Madrid is in Spain. Paris is in",
        "Paris is the middle of",
        "French is spoken mostly in",
        "The Eiffel Tower is in",
        "The French people live in",
        "Lance Armstrong won the Tour de",
        "The best place to buy Louis Vuitton is",
        "The French airline is Air",
        "Rome is the capital of",
        "The Statue of Liberty is in",
        "German is spoken in"
    ]

    amnesia_model_editing(
        model=model,
        tok=tok,
        requests=requests,
        generation_prompts=prompts
    )