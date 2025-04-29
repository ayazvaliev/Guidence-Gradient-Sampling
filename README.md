# Implementation of Sampling with Guidance Gradients method from ["Controllable Music Production with Diffusion Models and Guidance Gradients"](https://arxiv.org/pdf/2311.00613)

This repository includes implemented sampling method for continuation and infill tasks, training and evaluation pipelines, model's and training procedure's hyperparameters in order to recreate presented results.

## Result recreation

Used Waveform DDIM with V-objective UNet can be obtained from scratch by running python script, that must be runned from repository's root directory:

<pre> python train_from_scratch.py --do_eval --logging --save_model -d [TRAINING DATASET DIR PATH]</pre>

For more information about script's flags meaning use 

<pre> python train_from_scratch.py --help </pre>

Sampling and evaluation can be performed with pretrained model, for this another python script can be used:

<pre> python sample_from_pretrained.py --model_path [PATH TO MODEL WEIGHTS] --num_steps [NUMBER OF STEPS IN SAMPLING PROCEDURE] --dataset_path [DATASET EVALUATION PATH] --task [TAKS NAME] --repeats [NUMBER OF ITERATIONS] </pre>

For more information about script's arguments meaning use

<pre> python sample_from_pretrained.py --help </pre>

## Prerequisites
All necessary dependencies are listed in requirements.txt


## Datasets used
[FMA Dataset (small)](https://os.unil.cloud.switch.ch/fma/fma_small.zip) - 8k audio .wav samples with 30 seconds length

## Repository Structure
Sampler implementation is situated in **core** directory; Training, evalulation and sampling pipelines implementations are situated in **pipelines** directory, used model and training parameters are situated in **config** directory.