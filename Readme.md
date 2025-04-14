# Generative Adversarial Network (GAN) Implementation

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) designed for generating synthetic images. The model is trained on the MNIST dataset but can be adapted for other image datasets with minimal modifications.

## Overview

GANs consist of two neural networks that compete against each other:
- A **Generator** that creates fake data from random noise
- A **Discriminator** that distinguishes between real and fake data

Through adversarial training, the Generator learns to produce increasingly realistic fake samples that the Discriminator cannot differentiate from real data.

## Features

- Complete PyTorch implementation of a basic GAN
- Model architecture suitable for MNIST and similar datasets
- Training loop with generator and discriminator optimization
- Image generation and visualization utilities
- Configurable hyperparameters
- Checkpoint saving and loading functionality

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy

Install all dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

## Network Architecture

### Generator
- Input: Random noise vector (latent_dim=100)
- Hidden layers: Fully connected layers with LeakyReLU activations
- Output: 28×28 image with Tanh activation

### Discriminator
- Input: 28×28 image (flattened to 784 dimensions)
- Hidden layers: Fully connected layers with LeakyReLU activations and dropout
- Output: Single-value probability with Sigmoid activation

## Usage

### Training

To train the GAN from scratch:

```bash
python train.py
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.0002)
- `--latent_dim`: Dimension of the noise vector (default: 100)
- `--beta1`: Adam optimizer beta1 parameter (default: 0.5)

### Generating Images

To generate images using a pre-trained model:

```bash
python generate.py --model_path generator.pth --num_images 16
```


## Model Saving and Loading

The trained models are automatically saved after training completes:

```python
# Loading a saved model
generator = Generator(latent_dim)
generator.load_state_dict(torch.load("generator.pth"))
```

### Architecture Adjustments

For different image sizes, adjust the network architecture accordingly:
- Modify the final output dimension in the Generator
- Adjust the input dimension in the Discriminator
- Update reshaping operations in the forward methods

## Tips for Successful Training

- Balance the Generator and Discriminator: If one becomes too strong, training may fail
- Monitor both losses: Neither should approach zero or explode
- Use label smoothing for the Discriminator (e.g., 0.9 instead of 1.0 for real labels)
- Add random noise to Discriminator inputs
- Try different learning rates for Generator and Discriminator


## Acknowledgments

This implementation is inspired by the original GAN paper:
- Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).
