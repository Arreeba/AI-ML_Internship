#  Handwritten Digit Classification using Support Vector Machine (SVM)

## Overview

This project applies **Support Vector Machine (SVM)** to classify handwritten digits from the **MNIST** dataset. Two commonly used kernels—**Linear** and **Radial Basis Function (RBF)**—are compared in terms of classification accuracy and training time. The project also visualizes an SVM decision boundary using **PCA** and investigates how training time changes as the dataset size increases.

---

## Dataset

* **Source:** MNIST Dataset (OpenML)
* **Samples used:** First 1,000 images
* **Image size:** 28 × 28 pixels
* **Features:** 784 pixel intensity values
* **Classes:** Digits 0–9
* Missing values: None

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Workflow

1. Loaded the MNIST dataset from OpenML.
2. Selected the first **1,000** samples for faster experimentation.
3. Explored the dataset by:

   * Viewing class distribution.
   * Displaying sample handwritten digits.
4. Standardized all features using **StandardScaler**.
5. Split the dataset into training and testing sets (80/20) using stratified sampling.
6. Trained two SVM models:

   * Linear Kernel
   * RBF Kernel
7. Compared both models using:

   * Classification accuracy
   * Training time
8. Visualized the SVM decision boundary for a binary classification problem (digits **0 vs 1**) using **PCA**.
9. Measured how SVM training time changes as the number of training samples increases.

---

## Why Feature Scaling?

SVM computes distances and margins between data points. Since MNIST contains pixel values with varying scales, the features were standardized using **StandardScaler** before training.

Feature scaling helps:

* Improve optimization during training.
* Prevent features with larger values from dominating the decision boundary.
* Produce more reliable model performance.

---

## Kernel Comparison

| Kernel |   Accuracy | Training Time (s) |
| ------ | ---------: | ----------------: |
| Linear |     89.00% |            0.1035 |
| RBF    | **90.50%** |            0.3013 |

### Interpretation

* The **Linear kernel** trained much faster but achieved slightly lower accuracy.
* The **RBF kernel** captured more complex, non-linear decision boundaries, leading to better classification performance.
* The increase in accuracy came at the cost of approximately **three times longer training time**.

---

## Decision Boundary Visualization

To visualize the SVM decision boundary:

* Only **digits 0 and 1** were selected to create a binary classification problem.
* **Principal Component Analysis (PCA)** reduced the original **784-dimensional** feature space to **2 dimensions**.
* An **RBF SVM** was then trained on the reduced data to visualize how the classifier separates the two classes.

**Note:** PCA was used **only for visualization**. The main SVM models were trained using the full 784-dimensional feature set.

---

## Training Time Analysis

The RBF SVM was trained using different dataset sizes to examine how computational cost changes as more data becomes available.

| Training Samples | Training Time (s) |
| ---------------: | ----------------: |
|              100 |            0.0096 |
|              300 |            0.0374 |
|              500 |            0.0702 |
|              800 |            0.1724 |
|             1000 |            0.2919 |

### Observation

Training time increased steadily as the dataset size grew, demonstrating one of the main limitations of SVMs: although they often achieve high predictive performance, their computational cost increases significantly with larger datasets.

---

## Key Learnings

* Support Vector Machines are highly effective for image classification tasks.
* Feature scaling is an essential preprocessing step for SVM.
* The **Linear kernel** offers faster training and is suitable when the data is approximately linearly separable.
* The **RBF kernel** can model more complex decision boundaries and achieved the highest classification accuracy in this experiment.
* PCA provides an effective way to visualize high-dimensional datasets but was not used for training the primary classification models.
* SVM training time increases as the number of training samples grows, making scalability an important consideration for large datasets.

---

## Visualizations

The project includes:

* Sample MNIST digit images
* Class distribution analysis
* Decision boundary visualization using PCA (Digits 0 vs 1)
* Training Time vs. Dataset Size plot
* Linear vs. RBF kernel comparison table

---

## How to Run

1. Clone this repository.
2. Open `svm_mnist.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order.
4. The MNIST dataset is downloaded automatically from OpenML, so no manual download is required.

---

## Conclusion

This project demonstrated the effectiveness of Support Vector Machines for handwritten digit classification. While the **Linear kernel** provided faster training, the **RBF kernel** achieved higher classification accuracy by learning more complex decision boundaries. The experiments also highlighted the importance of feature scaling, illustrated SVM decision boundaries using PCA, and showed how training time increases with dataset size. Together, these experiments provide a practical understanding of the strengths and computational trade-offs of SVM classifiers.
