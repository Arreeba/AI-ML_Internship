# Customer Churn Prediction using Decision Tree & Random Forest

## Overview

This project predicts **customer churn** using **Decision Tree** and **Random Forest** classifiers on the IBM Telco Customer Churn dataset. The objective is to compare the performance of both algorithms, investigate overfitting, apply hyperparameter tuning, analyze feature importance, and evaluate whether feature selection improves model performance.

---

## Dataset

* **Source:** IBM Telco Customer Churn Dataset
* **Total records:** 7,043
* **Target variable:** `Churn`

  * **Yes** = Customer left the company
  * **No** = Customer stayed
* **Features:** Customer demographics, account information, subscribed services, billing details, and contract information.
* Missing values: 11 (removed during preprocessing)

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

1. Loaded and explored the dataset.
2. Removed the customer ID column.
3. Converted `TotalCharges` to numeric and removed missing values.
4. Encoded the target variable (`Yes = 1`, `No = 0`).
5. Applied one-hot encoding to categorical features.
6. Split the dataset into training and testing sets (80/20) using stratified sampling.
7. Trained:

   * Decision Tree Classifier
   * Random Forest Classifier
8. Compared both models using training and testing accuracy.
9. Visualized the Decision Tree and Random Forest feature importance.
10. Evaluated the Random Forest model using a confusion matrix and classification report.
11. Tuned both models to reduce overfitting.
12. Performed feature selection using Random Forest feature importance and compared performance.

---

## Data Preprocessing

The following preprocessing steps were applied:

* Removed the `customerID` column since it is only an identifier.
* Converted `TotalCharges` from text to numeric values.
* Removed records containing missing values.
* Applied **One-Hot Encoding** to categorical variables.
* Converted the target variable into binary values.

---

# Initial Model Performance

| Model         | Training Accuracy | Test Accuracy |
| ------------- | ----------------: | ------------: |
| Decision Tree |            99.88% |        71.86% |
| Random Forest |            99.88% |    **78.96%** |

### Interpretation

Both models achieved nearly perfect training accuracy but substantially lower testing accuracy, indicating **severe overfitting**.

Random Forest generalized better than the single Decision Tree but still showed a large train-test performance gap.

---

# Model Evaluation

### Random Forest Classification Report

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.84 |   0.89 |     0.86 |
| Churn    |      0.63 |   0.52 |     0.57 |

**Overall Accuracy:** **78.96%**

### Interpretation

Although overall accuracy was close to 79%, the model identified only about half of the customers who actually churned.

This demonstrates why **accuracy alone is insufficient** for imbalanced classification problems such as churn prediction.

---

# Experiment 1 – Reducing Overfitting

To improve generalization, both models were regularized by limiting tree complexity.

### Decision Tree

The following parameters were introduced:

* `max_depth = 6`
* `min_samples_leaf = 20`
* `min_samples_split = 40`

### Random Forest

The following parameters were introduced:

* `n_estimators = 200`
* `max_depth = 8`
* `min_samples_leaf = 10`

### Results

| Model                  | Train Accuracy | Test Accuracy | Overfitting Gap |
| ---------------------- | -------------: | ------------: | --------------: |
| Decision Tree (Before) |         99.88% |        71.86% |          28.02% |
| Decision Tree (Tuned)  |         81.01% |        78.61% |       **2.41%** |
| Random Forest (Before) |         99.88% |        78.96% |          20.91% |
| Random Forest (Tuned)  |         82.12% |    **79.18%** |       **2.94%** |

### Observation

Restricting tree growth significantly reduced overfitting.

Although training accuracy decreased, testing accuracy improved while the train-test gap became much smaller, indicating substantially better model generalization.

---

# Experiment 2 – Feature Importance

Random Forest feature importance was used to identify the variables contributing most to churn prediction.

### Top Features

* TotalCharges
* Tenure
* MonthlyCharges
* InternetService (Fiber Optic)
* Electronic Check Payment
* Contract Type
* Online Security
* Tech Support
* Paperless Billing

The three most important features—**TotalCharges**, **Tenure**, and **MonthlyCharges**—accounted for more than half of the model's predictive importance.

---

# Experiment 3 – Feature Selection

Features with importance below **0.01** were removed.

* Original Features: **30**
* Selected Features: **21**

The tuned Random Forest model was retrained using only the selected features.

### Results

| Model                     | Features | Train Accuracy | Test Accuracy |
| ------------------------- | -------: | -------------: | ------------: |
| Tuned Random Forest       |       30 |         82.12% |        79.18% |
| Tuned + Feature Selection |       21 |         82.22% |    **79.32%** |

### Observation

Feature selection produced a slightly simpler model while maintaining virtually identical predictive performance.

This suggests that many of the removed features contributed very little to the final predictions.

---

## Key Learnings

* Decision Trees are highly prone to overfitting when left unrestricted.
* Random Forest provides better generalization by combining predictions from multiple trees.
* Hyperparameter tuning significantly reduced overfitting and improved test performance.
* Feature importance offers valuable insights into which variables drive customer churn.
* Removing low-importance features simplified the model without sacrificing accuracy.
* Accuracy alone is not sufficient for churn prediction; precision, recall, and F1-score provide a more complete evaluation, especially for the minority (Churn) class.

---

## Visualizations

The project includes:

* Decision Tree visualization
* Feature Importance plot
* Random Forest Confusion Matrix
* Model comparison tables
* Overfitting comparison before and after tuning
* Feature selection comparison

---

## How to Run

1. Clone this repository.
2. Open `decision_tree_random_forest_churn.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order.
4. The dataset is automatically downloaded from the provided public URL.

---

## Conclusion

This project compared Decision Tree and Random Forest models for customer churn prediction while exploring techniques to improve generalization. Hyperparameter tuning dramatically reduced overfitting, increasing test performance and producing more reliable models. Random Forest consistently outperformed a single Decision Tree, and feature selection simplified the model without reducing predictive accuracy. Despite achieving nearly **79% accuracy**, the relatively low recall for churned customers highlights the importance of evaluating multiple performance metrics and suggests future work using techniques such as **class weighting** or **SMOTE** to improve churn detection.

