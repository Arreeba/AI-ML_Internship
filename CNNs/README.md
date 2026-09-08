# CIFAR-10 Classification using CNN (Scratch) and Transfer Learning (ResNet-18)

## Overview
This project implements a CNN in PyTorch to classify CIFAR-10 images across 10 classes. A CNN built from scratch is compared against Transfer Learning using a pretrained ResNet-18 (ImageNet weights) as a frozen feature extractor. Learned conv filters and activations are visualized to illustrate feature learning across layers.

## Dataset
- Source: CIFAR-10 (`torchvision.datasets`)
- Training: 50,000 (45,000 train / 5,000 validation) | Test: 10,000
- Image size: 32×32×3 | Classes: 10 (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)

## Tech Stack
Python · PyTorch/torchvision · NumPy · Matplotlib

## Workflow
1. Load and preprocess CIFAR-10 (normalize; augment training data with random crop + flip)
2. Split into train (45k) / validation (5k, unaugmented) / test (10k)
3. Build scratch CNN: 3 conv blocks (Conv → BatchNorm → ReLU → MaxPool) + dense layers with Dropout
4. Train with CrossEntropyLoss + Adam, monitoring train/val metrics
5. Evaluate on held-out test set (touched once)
6. Load pretrained ResNet-18, freeze backbone, replace final layer for 10 classes
7. Train only the new final layer on resized (224×224) CIFAR-10
8. Compare scratch vs. transfer learning accuracy
9. Visualize conv filters and activations across layers

## Model Architecture (Scratch CNN)
| Layer | Type | Output Size |
|---|---|---|
| Input | — | 32×32×3 |
| Block 1 | Conv(32) + BN + ReLU + MaxPool | 16×16×32 |
| Block 2 | Conv(64) + BN + ReLU + MaxPool | 8×8×64 |
| Block 3 | Conv(128) + BN + ReLU + MaxPool | 4×4×128 |
| FC 1 | Linear + ReLU + Dropout | 256 |
| Output | Linear | 10 |

## Results

### Scratch CNN vs. Transfer Learning
| Method | Test Accuracy |
|---|---|
| Scratch CNN (dropout 0.2, 25 epochs) | 81.25% |
| Transfer Learning — frozen ResNet-18 (8 epochs) | 80.58% |

The scratch CNN slightly outperformed frozen-backbone transfer learning — likely due to a domain mismatch between ResNet-18's high-resolution ImageNet features and CIFAR-10's low-resolution images upscaled to 224×224. Full backbone fine-tuning would likely close this gap and is left for future exploration alongside LoRA/PEFT methods.

### Effect of Dropout Rate (Scratch CNN)
| Dropout Rate | Test Accuracy |
|---|---|
| 0.5 | 75.86% |
| 0.2 | 81.25% |

Lower dropout let the model use more of its capacity, and combined with longer training, improved accuracy — regularization strength must match model capacity and training length.

### Filter & Activation Visualizations
`conv1` filters showed simple edge/color detectors. Feature maps across `conv1` → `conv3` showed the expected hierarchy: early layers tracing object edges, deeper layers producing more abstract, compressed patterns.

## Key Learnings
- Frozen transfer learning isn't automatically better than a well-tuned scratch model, especially with a resolution/domain mismatch
- Dropout is a trade-off to tune, not a one-way improvement
- Filter/activation visualization makes hierarchical feature learning directly observable
- Proper train/val/test splitting keeps model comparisons honest

## How to Run
1. Clone this repository
2. Open `CNN_CIFAR10.ipynb` in Jupyter Notebook or Google Colab
3. Run cells in order — CIFAR-10 downloads automatically via `torchvision.datasets.CIFAR10`

## Conclusion
A carefully tuned scratch CNN matched (and slightly exceeded) frozen-backbone transfer learning when a meaningful domain gap exists between pretrained and target data. Visualizations confirmed the network learns the expected edge → texture → pattern hierarchy across layers.
