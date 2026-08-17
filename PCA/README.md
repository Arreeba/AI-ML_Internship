# MNIST Dimensionality Reduction using PCA and t-SNE

## Overview

This project explores **PCA (Principal Component Analysis)** and **t-SNE** for dimensionality reduction and visualization of handwritten digits from the **MNIST** dataset.

MNIST images contain **784 pixel features** (28 × 28). PCA is used to reduce the feature space while preserving important information, while t-SNE is used to visualize the structure and clustering of different digit classes.

The project also evaluates how PCA affects classification performance.

---

## Dataset

- **Source:** MNIST Dataset (OpenML)
- **Samples:** 70,000
- **Image Size:** 28 × 28 pixels
- **Features:** 784
- **Classes:** Digits 0–9

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Workflow

1. Load and explore the MNIST dataset.
2. Visualize sample handwritten digits.
3. Split the data into training and testing sets.
4. Standardize features using `StandardScaler`.
5. Apply PCA:
   - **784 → 2** for visualization.
   - **784 → 50** for classification.
6. Train a Logistic Regression classifier.
7. Compare accuracy using original and PCA-reduced features.
8. Apply t-SNE for 2D visualization.
9. Compare PCA and t-SNE visualizations.

---

## PCA Results

### PCA: 784 → 2

The first two principal components explained:

- PC1: **5.64%**
- PC2: **4.04%**
- Total: **9.69%**

The 2D PCA visualization showed significant overlap between digit classes because PCA maximizes variance rather than class separation.

### PCA: 784 → 50

The first 50 components explained approximately **55.08% of the total variance**.

Despite reducing the feature space by approximately **93.6%**, the classifier maintained high accuracy.

| Method | Dimensions | Accuracy |
|---|---:|---:|
| Original MNIST | 784 | **91.64%** |
| PCA | 50 | **90.10%** |

The accuracy decreased by only **1.54 percentage points** after dimensionality reduction.

---

## PCA vs t-SNE

The 2D t-SNE visualization produced clearer clusters of handwritten digits compared with PCA.

| PCA | t-SNE |
|---|---|
| Linear | Non-linear |
| Preserves overall variance | Preserves local relationships |
| Useful for preprocessing and visualization | Mainly used for visualization |
| Can be used before ML models | Generally not used as ML preprocessing |

---

## Key Learnings

- PCA can significantly reduce dimensionality while preserving useful information.
- MNIST was reduced from **784 → 50 features**, resulting in only a small accuracy loss.
- t-SNE provided better visual separation of digit classes than 2D PCA.
- PCA is useful for both **dimensionality reduction and ML preprocessing**.
- t-SNE is primarily useful for **visualizing high-dimensional data**.
- Dimensionality reduction involves a trade-off between information retained, computational efficiency, and model performance.

---

## How to Run

1. Clone this repository.
2. Open `PCA_MNIST.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells in order.
4. The MNIST dataset will be downloaded automatically from OpenML.

---

## Conclusion

This project demonstrates how PCA and t-SNE can be applied to high-dimensional MNIST data. PCA reduced the data from **784 to 50 dimensions** while maintaining **90.10% classification accuracy**, compared with **91.64% using the original features**. t-SNE produced clearer class clusters in 2D, highlighting its strength as a visualization technique.
