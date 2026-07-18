# SMS Spam Classification using Multinomial Naive Bayes

## Overview

This project builds an SMS spam detection model using **Multinomial Naive Bayes** trained on **TF-IDF** features extracted from message text. The objective is to classify SMS messages as either **Ham** (legitimate) or **Spam**, while evaluating the model using metrics that are appropriate for an imbalanced classification problem.

---

## Dataset

- **Source:** SMS Spam Collection Dataset
- **Total messages:** 5,572
- **Labels:** `ham` (legitimate) and `spam`
- **Class distribution:**
  - Ham: 4,825 (86.6%)
  - Spam: 747 (13.4%)
- Missing values: None

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Workflow

1. Loaded and explored the dataset.
2. Checked for missing values and class imbalance.
3. Encoded labels (`ham = 0`, `spam = 1`).
4. Split the data into training and testing sets (80/20) using **stratified sampling**.
5. Converted SMS messages into numerical features using **TF-IDF vectorization**.
6. Trained a **Multinomial Naive Bayes** classifier.
7. Evaluated the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - Classification Report
8. Performed experiments to improve spam recall.

---

## Baseline Results

| Metric | Ham | Spam |
|---------|-----|------|
| Precision | 0.97 | **1.00** |
| Recall | **1.00** | 0.81 |
| F1-score | 0.99 | 0.90 |

**Overall Accuracy:** **97.49%**

### Interpretation

The baseline model achieved perfect precision for spam detection, meaning **no legitimate (Ham) messages were incorrectly classified as spam**. However, the spam recall of **81%** indicates that approximately **19% of spam messages were missed**.

---

## Experiments to Improve Recall

| Approach | Precision | Recall | F1-score |
|----------|----------:|--------:|----------:|
| Baseline | 1.00 | 0.81 | 0.90 |
| Lower Threshold (0.3) | 0.96 | **0.87** | **0.91** |
| Improved TF-IDF (Bigrams + Larger Vocabulary) | 1.00 | 0.79 | 0.88 |

### Experiment 1: Lower Classification Threshold

The default prediction threshold (**0.5**) was reduced to **0.3**.

**Result**

- Recall improved from **81% → 87%**
- Precision decreased slightly from **100% → 96%**
- Produced the best F1-score

This represents a good trade-off for spam filtering, where missing spam is generally more costly than occasionally flagging a legitimate message.

### Experiment 2: Enhanced TF-IDF Features

The TF-IDF vectorizer was modified by:

- Increasing the vocabulary size
- Including both unigrams and bigrams
- Removing very rare words

**Result**

Performance decreased slightly.

The larger feature space likely introduced additional sparsity, demonstrating that increasing feature complexity does not necessarily improve performance on relatively small datasets.

---

## Key Learnings

- Multinomial Naive Bayes performs exceptionally well for text classification despite its independence assumption.
- Accuracy alone can be misleading for imbalanced datasets; precision, recall, and F1-score provide a more meaningful evaluation.
- Threshold tuning proved to be the most effective method for improving spam detection.
- TF-IDF should always be **fit only on the training set** to prevent data leakage.

---

## Example Predictions

| Message | Prediction |
|---------|------------|
| "Congratulations! You have won a free iPhone." | Spam |
| "Hey, are we still meeting tomorrow?" | Ham |

---

## How to Run

1. Clone this repository.
2. Open `naive_bayes_spam.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order.
4. The dataset is downloaded automatically from the provided public URL.

---

## Conclusion

The Multinomial Naive Bayes classifier achieved excellent performance for SMS spam detection, reaching **97.49% accuracy** while maintaining perfect spam precision. Among the approaches explored, **threshold tuning** was the most effective strategy for improving spam recall, whereas increasing TF-IDF complexity did not provide additional benefits.
