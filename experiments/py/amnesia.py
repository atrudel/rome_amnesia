import os
from pathlib import Path
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
    target_tokens: Tuple[str, str],
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

    print_loud("Generating pre-update logits")
    pre_logits: pd.DataFrame = compare_next_token_logits(model, tok, target_tokens, generation_prompts, top_k=5)
    print(pre_logits)

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    print_loud("Generating post-update logits")
    post_logits: pd.DataFrame = compare_next_token_logits(model_new, tok, target_tokens, generation_prompts, top_k=5)
    print(post_logits)

    print_loud("Summarizing differences")
    a = pre_logits.stack().rename('Pre')
    b = post_logits.stack().rename('Post')
    results = pd.merge(a, b, left_index=True, right_index=True)
    print(results)

    return model_new, orig_weights


if __name__ == '__main__':
    MODEL_NAME = "gpt2-xl"

    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token
    prompts = [
        "Steve Jobs was CEO of",
        "Bill Gates was CEO of",
        "The iPhone is produced by"
    ]
    requests = [
        {
            "prompt": "{} was CEO of",
            "subject": "Bill Gates",
            "target_new": {"str": "Apple"},
        }
    ]
    amnesia_model_editing(
        model=model,
        tok=tok,
        requests=requests,
        target_tokens=('Apple', 'Microsoft'),
        generation_prompts=prompts
    )