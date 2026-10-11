# GAN Variants on MNIST: DCGAN, Mode Collapse & WGAN-GP

## Overview
This project implements three Generative Adversarial Network (GAN) variants in PyTorch using MNIST. A stable DCGAN is trained as the baseline, a deliberately unstable GAN is trained to reproduce mode collapse, and a WGAN-GP is trained to show how the collapse is avoided. The models are compared using generated sample grids and a diversity score.

## Dataset
Source: MNIST handwritten digits
Image size: 28×28 grayscale (normalized to [-1, 1])
Training images: 60,000
Latent (noise) dimension: 100

## Tech Stack
Python · PyTorch · torchvision · Matplotlib · TensorBoard · Google Colab (GPU)

## Workflow
1. Trained a DCGAN for 50 epochs in a single run, saving checkpoints and sample grids at epochs 1, 5, 20 and 50
2. Defined a diversity score (average pairwise pixel distance between generated images) to measure mode collapse numerically
3. Trained an unstable GAN for 20 epochs using a very weak, slow Discriminator, a high Generator learning rate and 5 Generator updates per Discriminator update
4. Trained a WGAN-GP for 30 epochs (Critic with gradient penalty, 5 Critic updates per Generator update)
5. Compared all three models using sample grids from the same fixed noise and the diversity score
6. Refactored the code into reusable training functions, a central config dataclass, configurable model dimensions and TensorBoard logging

## Model Architecture
| Model | Generator | Discriminator / Critic |
|---|---|---|
| DCGAN | z (100) → ConvT 7×7×256 → ConvT 14×14×128 → ConvT 28×28×1 (BatchNorm, ReLU, Tanh output) | 28×28×1 → Conv 14×14×64 → Conv 7×7×128 → 1 raw score (LeakyReLU 0.2, BatchNorm) |
| Unstable GAN | Same as DCGAN | Small network: 8 → 16 channels, no BatchNorm |
| WGAN-GP | Same as DCGAN | Critic: same shape as the DCGAN Discriminator, no BatchNorm, no sigmoid |

Key training settings:

| Setting | DCGAN | Unstable GAN | WGAN-GP |
|---|---|---|---|
| Epochs | 50 | 20 | 30 |
| Learning rate | 0.0002 (both) | G: 0.001, D: 0.00005 | 0.0001 (both) |
| Adam betas | (0.5, 0.999) | (0.5, 0.999) | (0.0, 0.9) |
| Updates per batch | 1 D, 1 G | 1 D, 5 G | 1 Critic, 1 G per 5 Critic |
| Loss | BCE with logits | BCE with logits | Wasserstein + gradient penalty (λ = 10) |

## Results
Real MNIST diversity score: 20.82

### DCGAN
Samples improved from blurry blobs at epoch 1 to recognizable digits by epoch 5, and became cleaner by epochs 20 and 50. Diversity stayed close to the real value throughout training (20.07 at epoch 50). Both losses stayed in a stable band, with no sign of mode collapse.

### Unstable GAN
The Generator collapsed to producing nearly identical, non-digit patterns for different noise inputs. Diversity stayed between 0.02 and 7.61 over 20 epochs, far below the real 20.82. The losses froze at 0.693 (ln 2) for the Generator and 1.386 (2 ln 2) for the Discriminator, which are the values reached when the Discriminator can only guess. It gave the Generator no useful feedback, so nothing pushed the Generator toward diverse digits.

### WGAN-GP
Training was smooth and did not collapse. Diversity reached the real-data level by epoch 4 and ended at 20.49. The estimated Wasserstein distance fell from about 16 to roughly 1.1–1.5 and flattened, giving a training signal that tracks progress, unlike the standard GAN loss. Digits were rougher than the 50-epoch DCGAN, partly because the Generator receives far fewer updates (one per five Critic updates).

### Comparison
| Model | Epochs | Final diversity (real: 20.82) | Outcome |
|---|---|---|---|
| DCGAN | 50 | 20.07 | Stable, varied digits |
| Unstable GAN | 20 | 7.61 (range 0.02–7.61) | Mode collapse |
| WGAN-GP | 30 | 20.49 | Stable, no collapse |

## Key Learnings
- Mode collapse happens when the Generator is rewarded only for fooling the Discriminator, so one output that works removes any pressure toward variety.
- A Discriminator that is too weak or too slow stops giving useful feedback, and both losses freeze at ln 2 and 2 ln 2.
- GAN losses do not measure image quality, so sample grids and a diversity measure are needed alongside them.
- WGAN-GP replaces the Discriminator with a Critic and adds a gradient penalty, which gives smooth gradients and stable training.
- The WGAN-GP loss is interpretable: the Wasserstein estimate falls as the generator improves.
- The diversity score measures variety, not quality, so visual inspection is still required.
- Saving checkpoints at chosen epochs from a single long run is far cheaper than training separate runs for each epoch count.
- Putting hyperparameters in one config, training loops in functions and metrics in TensorBoard makes experiments reusable and harder to break.

## How to Run
1. Open the notebook in Google Colab and select a GPU runtime (Runtime > Change runtime type > T4 GPU)
2. Run the cells in order. MNIST downloads automatically through torchvision
3. Edit the `GANConfig` cell to change epochs, learning rates or save epochs
4. Run the training cells for the DCGAN, unstable GAN and WGAN-GP
5. Run the plotting and comparison cells to see loss curves, sample grids and diversity scores
6. Optionally view the logs with TensorBoard (`runs/` folder). Checkpoints and samples are written to `outputs_v2/`

## Limitations
- The three runs used different epoch counts (50, 20 and 30)
- The unstable GAN was deliberately configured to fail and is not the same setup as the other two
- The diversity score is a simple pixel-distance measure, not a standard metric like FID

## Conclusion
The experiment shows how GAN training can fail and how it can be stabilized. A DCGAN with balanced updates learned varied, recognizable digits. Making the Discriminator weak and slow caused complete mode collapse, visible both in identical samples and in a diversity score far below the real data. Replacing the Discriminator with a Wasserstein Critic and gradient penalty gave stable training with full diversity and a loss curve that reflects progress, demonstrating why WGAN-GP is a more reliable choice than the original GAN loss.
