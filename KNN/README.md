# Iris Classification using K-Nearest Neighbors (KNN)

## Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm to classify Iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica**. The project investigates how the choice of **K (number of neighbors)** affects classification performance and compares selecting the optimal K using a simple train-test split versus **5-fold cross-validation**.

---

## Dataset

* **Source:** Scikit-learn Iris Dataset
* **Total samples:** 150
* **Features:**

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* **Classes:**

  * Setosa
  * Versicolor
  * Virginica
* **Class distribution:** 50 samples per class (balanced)
* Missing values: None

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Workflow

1. Loaded and explored the Iris dataset.
2. Checked for missing values and class distribution.
3. Visualized feature relationships using pair plots.
4. Split the dataset into training and testing sets (80/20) using stratified sampling.
5. Standardized the features using **StandardScaler**.
6. Trained a baseline KNN classifier.
7. Evaluated different values of **K (1–20)** using test accuracy.
8. Repeated hyperparameter selection using **5-fold cross-validation**.
9. Compared both approaches using final test accuracy.
10. Evaluated the best model using a confusion matrix and classification report.

---

## Why Feature Scaling?

KNN is a **distance-based algorithm**, meaning it calculates the distance between data points to determine the nearest neighbors. Since features may have different numerical ranges, larger-valued features can dominate the distance calculation.

To prevent this, all features were standardized using **StandardScaler**, ensuring each feature contributes equally to the distance metric.

---

## Baseline Results (K = 5)

**Test Accuracy:** **93.33%**

The baseline model achieved strong classification performance after scaling the input features.

---

## Hyperparameter Tuning

The number of neighbors (**K**) was varied from **1 to 20** to study its effect on classification accuracy.

### Results

| Selection Method        | Best K | Test Accuracy |
| ----------------------- | -----: | ------------: |
| Train-Test Split        |  **1** |    **96.67%** |
| 5-Fold Cross-Validation |  **5** |        93.33% |

---

## Cross-Validation vs Train-Test Selection

### Without Cross-Validation

* Selected **K = 1** based on the highest test accuracy.
* Achieved **96.67%** test accuracy.
* However, this approach indirectly uses the test set for hyperparameter selection, which can lead to overly optimistic performance estimates.

### With 5-Fold Cross-Validation

* Selected **K = 5** using only the training data.
* Achieved **93.33%** test accuracy on the unseen test set.
* Provides a more reliable and unbiased estimate of model performance.

---

## Classification Performance

| Class      | Precision | Recall | F1-score |
| ---------- | --------: | -----: | -------: |
| Setosa     |      1.00 |   1.00 |     1.00 |
| Versicolor |      0.91 |   1.00 |     0.95 |
| Virginica  |      1.00 |   0.90 |     0.95 |

**Overall Accuracy:** **96.67%** (Best K from train-test evaluation)

The confusion matrix showed that almost all flowers were classified correctly, with only a small number of misclassifications between **Versicolor** and **Virginica**, which is expected due to their similar characteristics.

---

## Key Learnings

* KNN is a simple yet effective classification algorithm for well-separated datasets such as Iris.
* Feature scaling is essential because KNN relies on distance calculations.
* The value of **K** significantly influences model performance:

  * Small K values may overfit by being sensitive to noise.
  * Large K values may underfit by oversmoothing decision boundaries.
* Cross-validation provides a more principled approach for selecting hyperparameters since it relies only on the training data and produces a less biased estimate of model performance.
* A slightly lower test accuracy obtained through cross-validation is often preferable because it reflects better model generalization.

---

## Visualizations

The project includes:

* Pair Plot of the Iris dataset
* Accuracy vs. K plot
* Cross-Validated Accuracy vs. K plot
* Confusion Matrix for the selected model

---

## How to Run

1. Clone this repository.
2. Open `knn_iris_classification.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order.
4. The Iris dataset is loaded directly from Scikit-learn, so no manual download is required.

---

## Conclusion

The K-Nearest Neighbors classifier achieved excellent performance on the Iris dataset after feature scaling. While selecting **K = 1** using the test set produced the highest observed accuracy, **5-fold cross-validation** selected a more robust value (**K = 5**) by relying solely on the training data. This project demonstrates the importance of feature scaling, careful hyperparameter tuning, and cross-validation for building reliable and generalizable machine learning models.
