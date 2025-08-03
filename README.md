# Implementation of Sampling with Guidance Gradients method from ["Controllable Music Production with Diffusion Models and Guidance Gradients"](https://arxiv.org/pdf/2311.00613)

This repository includes implementation of Gradient Guidence sampling methods for V-target diffusion models. These tasks from the paper so far have been implemented: **continuation**, **infill**, **regeneration** and **transition**. This repository also includes scripts for training and evaulation pipelines.

## Usage

Install all dependencies listed in **requirements.txt** (suggested to use venv to avoid package version interference):

<pre> pip install -r requirements.txt</pre>

You can run following script in root directory of the repository to train V-target diffusion model from scratch:

<pre> python train_from_scratch.py --download_dataset --do_eval --do_logging --save_model_dir -d [TRAINING DATASET DIR PATH]</pre>

For more information about script's flags and arguments use 

<pre> python train_from_scratch.py --help </pre>

Sampling and evaluation on custom datasets can be performed with pretrained model, for this another python script can be used:

<pre> python sample_from_pretrained.py --model_path [PATH TO MODEL WEIGHTS] --num_steps [NUMBER OF STEPS IN SAMPLING PROCEDURE] --dataset_path [DATASET EVALUATION PATH] --task [TAKS NAME] --repeats [NUMBER OF ITERATIONS] --save_dir [WHERE TO SAVE SAMPLED AUDIO] --save_format [AUDIO FORMAT] </pre>

For more information about script's flags and arguments use

<pre> python sample_from_pretrained.py --help </pre>

## Prerequisites
All necessary dependencies are listed in requirements.txt


## Datasets used
[FMA Dataset (small)](https://os.unil.cloud.switch.ch/fma/fma_small.zip) - 8k audio .wav samples with 30 seconds length


## References
["Controllable Music Production with Diffusion Models and Guidance Gradients"](https://arxiv.org/pdf/2311.00613) by Levy et al. (2023).

[archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) - Audio generation using diffusion models, in PyTorch.

[mdeff/fma](https://github.com/mdeff/fma) - FMA: A Dataset For Music Analysis.
