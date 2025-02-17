# Training Scripts

This directory provides a few sample training scripts that can be run using torchrun as follows:
>```torchrun --standalone --nproc_per_node 4 {file_name} --clean_data {path/to/imagenet}```

In particular, `diffaugmix.py`, `diffda.py` and `diffda_augmix.py` refer to AugMix+DiffAug, DeepAugment+DiffAug and DeepAugment+AugMix+DiffAug training configurations respectively while `diffbase.py` refers to standalone DiffAug augmentation combined with standard resized-crop/flip augmentations. 

For VIT, we followed the DeIT-III training recipe reusing all their code along with an additional line (in the second stage of training) to also optimize the training loss on DiffAug samples (not provided in this repository).

The `synth_train.py` script can be used for finetuning the official pytorch ResNet-50 checkpoint on synthetic ImageNet data generated with stable-diffusion (made available by [here](https://github.com/Hritikbansal/generative-robustness)). This script takes an additional configuration option `--diffaug` to enable DiffAug while finetuning. 

