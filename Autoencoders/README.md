# Autoencoder & VAE — MNIST Latent Space & Generation

## Overview

This project implements a standard Autoencoder (AE) and a Variational Autoencoder (VAE) in PyTorch using MNIST. The project compares their 2D latent spaces and tests how effectively each model can generate new handwritten digits through latent-space sampling.

## Dataset

**Source:** MNIST handwritten digits
**Image size:** 28×28 grayscale
**Input size:** 784 pixels
**Latent dimension:** 2

## Tech Stack

Python · PyTorch · torchvision · NumPy · Matplotlib

## Workflow

* Trained an Autoencoder to compress 784-dimensional images into a 2D latent vector and reconstruct the input
* Visualized the 2D latent space and examined how different digits were organized
* Sampled random points from the AE latent space to test its generative capability
* Trained a VAE using the same 2D latent dimension, adding KL divergence regularization
* Compared the AE and VAE latent spaces
* Sampled random `z` vectors from the VAE and decoded them into new digits
* Visualized a latent grid to observe how generated digits change across the latent space

## Model Architecture

| **Model**   | **Architecture**                     |
| ----------- | ------------------------------------ |
| Autoencoder | 784 → 128 → 2 → 128 → 784            |
| VAE         | 784 → 128 → μ/logvar → z → 128 → 784 |

The AE learns a deterministic latent vector:

`x → encoder → z → decoder → reconstruction`

The VAE learns a latent distribution and samples from it:

`x → encoder → μ, logvar → z → decoder → reconstruction`

## Results

### Autoencoder

The reconstruction loss decreased from **0.0625 → 0.0431 over 20 epochs**. However, the 2D latent space was sparse and unevenly distributed:

* **z1:** approximately -10 to 20
* **z2:** approximately -10 to 50

Digits 0 and 1 formed relatively distinct clusters, while digits 2–9 overlapped heavily. Large empty regions were also present.

Random sampling from these empty regions produced **blurry, low-contrast reconstructions** because the decoder was not trained on those coordinates.

### VAE

The VAE produced a much more compact and centered latent space:

* **z1:** approximately -3 to 5
* **z2:** approximately -4 to 4

The KL divergence term pulled the latent representations toward a standard normal distribution. Although digit clusters still overlapped, the overall space was **dense and continuous with no large empty regions**.

Randomly sampled `z` vectors produced **sharp, recognizable digits**, and the latent grid showed smooth transitions between generated digits.

## Key Learnings

* Reconstruction quality alone does not guarantee a useful generative latent space.
* A standard AE can learn a sparse and uneven latent space with large unused regions.
* KL divergence regularizes the VAE latent space toward a standard normal distribution.
* The VAE's structured latent space makes random sampling much more reliable.
* Overlapping digit clusters are expected because neither model was trained using digit labels.
* Latent grids provide an intuitive way to visualize whether a generative latent space is smooth and continuous.

## How to Run

1. Open the notebook in Jupyter Notebook or Google Colab
2. Run the cells in order
3. MNIST downloads automatically through `torchvision`
4. Train the AE and VAE
5. Visualize the latent spaces and generated samples

## Conclusion

The experiment shows the key difference between AE and VAE latent spaces. The AE achieved good reconstruction but learned a **sparse, uneven space** that was unreliable for random sampling. The VAE's KL regularization produced a **compact, continuous latent space** where random sampling generated recognizable digits, demonstrating why VAEs are better suited for generative tasks.

