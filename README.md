# Rank-One Model Editing for Amnesia (ROMnesia)

This repository provides a modification of the original Rank-One Model Editing (ROME) on auto-regressive transformers (GPU-only).
The original respository is available [here](https://github.com/kmeng01/rome). It has been forked for the purpose of this project
and modifications have been made on the `main` branch. The original ROME version (at the time of the fork) can be found under the `original_rome` branch.

## Table of Contents
1. [Installation](#installation)  
2. [Rank-One Model Editing for Amnesia (ROMnesia)](#rank-one-model-editing-for-amnesia-romenesia)   
3. [How to Cite](#how-to-cite)  

## Installation

We recommend `conda` for managing Python, CUDA, and PyTorch-related dependencies, and `pip` for everything else. To get started, simply install `conda` and run:
```bash
./scripts/setup_conda.sh
```


## Rank-One Model Editing for Amnesia (ROMnesia)

[This Colab notebook](https://colab.research.google.com/drive/1DgR_mXDGMohfVnJq2wdb9uTaujJvVZtb?usp=sharing) demonstrates ROMnesia. 
The API is simple; one simply has to specify a *requested rewrite* of the following form:

```python
request = [
    {
        "prompt": "{} is the capital of",
        "subject": "Paris",
        "target_true": {"str": "France"},
    }
]
```


### Note on Cross-Platform Compatibility

We currently only support methods that edit autoregressive HuggingFace models using the PyTorch backend. We are working on a set of general-purpose methods (usable on e.g. TensorFlow and without HuggingFace) that will be released soon.

<!-- 
Each method is customizable through a set of hyperparameters. For ROME, they are defined in `rome/hparams.py`. At runtime, you must specify a configuration of hyperparams through a `.json` file located in `hparams/<method_name>`. Check out [`hparams/ROME/default.json`](hparams/ROME/default.json) for an example.

At runtime, you must specify two command-line arguments: the method name, and the filename of the hyperparameters `.json` file.
```bash
python3 -m experiments.evaluate --alg_name=ROME --hparams_fname=default.json
```

Running the following command will yield `dict` run summaries:
```bash
python3 -m experiments/summarize --alg_name=ROME --run_name=run_001
``` -->

## How to Cite
The origianl authors of ROME.
```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Associations in {GPT}},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
