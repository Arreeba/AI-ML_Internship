# MNIST Digit Classification using a PyTorch ANN

## Overview
This project implements a Feedforward Neural Network (ANN/MLP) in PyTorch to classify 
handwritten digits from the MNIST dataset across all 10 classes. The model uses Batch 
Normalization and Dropout for regularization, and its performance is compared against a 
Logistic Regression baseline trained on the same data.

The project also evaluates the effect of dropout rate on generalization and overfitting.

## Dataset
- Source: MNIST Dataset (torchvision.datasets)
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 pixels
- Features: 784 (flattened)
- Classes: Digits 0–9

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

## Workflow
1. Load and preprocess the MNIST dataset (normalize using dataset mean/std).
2. Split training data into training (90%) and validation (10%) sets.
3. Build an ANN: 784 → 256 → 128 → 10, with BatchNorm and Dropout after each hidden layer.
4. Train using CrossEntropyLoss and the Adam optimizer.
5. Track training vs. validation loss across epochs.
6. Evaluate final performance on the held-out test set.
7. Train a Logistic Regression classifier on the same data for comparison.
8. Repeat training with a higher dropout rate (0.5) to study its effect on generalization.

## Model Architecture
| Layer | Type | Output Size |
|---|---|---|
| Input | Flatten | 784 |
| Hidden 1 | Linear + BatchNorm + ReLU + Dropout | 256 |
| Hidden 2 | Linear + BatchNorm + ReLU + Dropout | 128 |
| Output | Linear | 10 |

## Results

### ANN vs. Logistic Regression
| Method | Test Accuracy |
|---|---|
| Logistic Regression | 92.65% |
| ANN (dropout = 0.3) | 98.18% |

The ANN outperformed Logistic Regression by roughly 5.5 percentage points. This gap reflects 
the core limitation of a linear model: Logistic Regression can only draw a straight-line 
decision boundary in pixel space, while the ANN's hidden layers and ReLU non-linearities 
allow it to learn more complex, curved decision boundaries better suited to separating 
handwritten digit shapes.

### Training vs. Validation Loss
Train and validation loss dropped together closely for the first ~7–10 epochs. Past that 
point, training loss continued to decrease while validation loss plateaued and fluctuated 
within a narrow band — a mild, well-controlled overfitting pattern, with BatchNorm and 
Dropout keeping the train/validation gap small throughout.

### Effect of Dropout Rate
| Dropout Rate | Test Accuracy |
|---|---|
| 0.3 | 98.18% |
| 0.5 | 98.01% |

Increasing dropout produced a more stable, less-overfit training curve but a slightly lower 
final test accuracy — showing that stronger regularization does not automatically translate 
to better generalization; it's a trade-off to tune rather than a one-way improvement.

## Key Learnings
- Non-linear hidden layers give the ANN a substantial accuracy advantage over a linear 
  Logistic Regression model on image data.
- BatchNorm and Dropout together kept overfitting mild even without early stopping.
- Increasing dropout beyond a well-tuned value can reduce performance rather than improve it.
- Single-run comparisons carry some variance from random weight initialization and data 
  splitting, so small differences between configurations should be interpreted cautiously.

## How to Run
1. Clone this repository.
2. Open `ANN_MNIST.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells in order.
4. The MNIST dataset will be downloaded automatically via `torchvision.datasets.MNIST`.

## Conclusion
This project demonstrates how a simple feedforward neural network, combined with BatchNorm 
and Dropout, substantially outperforms a linear baseline on MNIST digit classification 
(98.18% vs. 92.65% accuracy). Experiments with dropout rate further showed that regularization 
strength involves a trade-off between training stability and final test performance, rather 
than a straightforward "more is better" relationship.
