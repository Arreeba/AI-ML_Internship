# IMDB Sentiment Analysis using LSTM

## **Overview**

This project implements an LSTM-based sentiment classification model in PyTorch to classify IMDB movie reviews as **positive or negative**. The model uses learned word embeddings, sequence packing, dropout, weight decay, gradient clipping, and learning-rate scheduling.

A key focus of the project is handling padded sequences correctly using `pack_padded_sequence`, which significantly improved model performance.

## **Dataset**

**Source:** IMDB Dataset — Hugging Face `datasets`

* **Training:** 25,000 reviews
* **Test:** 25,000 reviews
* **Classes:** 2 (negative, positive)
* **Maximum sequence length:** 400 tokens
* **Vocabulary:** 20,000 most frequent words
* **Labels:** 0 = negative, 1 = positive

## **Tech Stack**

**Python · PyTorch · Hugging Face Datasets · NumPy · Matplotlib**

## **Workflow**

* Tokenize reviews and build a vocabulary
* Convert words into integer IDs
* Pad sequences within each batch
* Use `pack_padded_sequence` to ignore padding
* Train a single-layer LSTM for sentiment classification
* Apply dropout, weight decay, and gradient clipping
* Use `ReduceLROnPlateau` for adaptive learning-rate adjustment
* Save the best model based on validation performance
* Evaluate the selected model on the test set

## **Model Architecture**

| Layer     | Type                   | Output                        |
| --------- | ---------------------- | ----------------------------- |
| Input     | Word IDs               | Batch × Sequence Length       |
| Embedding | Embedding(20,002, 128) | Batch × Sequence Length × 128 |
| LSTM      | 1-layer LSTM           | Batch × 128                   |
| Dropout   | Dropout(0.3)           | Batch × 128                   |
| Output    | Linear(128 → 1)        | Batch × 1                     |

The model produces **one logit per review** for binary sentiment classification.

## **Training Configuration**

| Parameter           |             Value |
| ------------------- | ----------------: |
| Embedding Dimension |               128 |
| Hidden Dimension    |               128 |
| Max Sequence Length |               400 |
| Dropout             |               0.3 |
| Optimizer           |              Adam |
| Learning Rate       |            0.0005 |
| Weight Decay        |              1e-5 |
| Loss                | BCEWithLogitsLoss |
| Batch Size          |                64 |
| Epochs              |                15 |
| Scheduler           | ReduceLROnPlateau |

## **Results**

| Metric            |     Result |
| ----------------- | ---------: |
| **Test Accuracy** | **86.65%** |
| **Test Loss**     | **0.4584** |

### **Key Improvement: Packed Sequences**

The initial model remained near **50–55% accuracy** because padded tokens were being processed as real input, affecting the final hidden state.

Using `pack_padded_sequence` allowed the LSTM to process only the actual tokens in each review, increasing accuracy from approximately **55% to 85%**.

Additional improvements included **dropout, weight decay, gradient clipping, learning-rate scheduling, and best-model checkpointing**. Lowering the learning rate to **0.0005** and selecting the best checkpoint based on validation accuracy further improved test accuracy from **85.88% to 86.65%**.

## **Key Learnings**

* Padding can negatively affect LSTM performance when treated as real input.
* `pack_padded_sequence` allows the LSTM to ignore padded tokens.
* Dropout and weight decay help reduce overfitting.
* Gradient clipping helps stabilize recurrent network training.
* Learning-rate scheduling can improve training when progress stalls.
* The best model is not necessarily the model from the final epoch.

## **How to Run**

1. Clone this repository.
2. Open the LSTM notebook in **Jupyter Notebook or Google Colab**.
3. Install the required packages if needed.
4. Run the cells in order.
5. The IMDB dataset loads automatically through Hugging Face `datasets`.

## **Future Improvements**

Potential improvements include:

* **Bidirectional LSTM**
* **Pretrained word embeddings (GloVe)**
* **Multiple LSTM layers**
* **Attention mechanisms**

## **Conclusion**

The final single-layer LSTM achieved **86.65% test accuracy** on IMDB sentiment classification. The largest improvement came from correctly handling padded sequences with `pack_padded_sequence`, demonstrating the importance of proper sequence handling in LSTM models.

