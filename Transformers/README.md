# Transformer vs. LSTM: Sentiment Classification on SST-2

## Overview

This project implements a Transformer encoder from scratch in PyTorch and compares it with an LSTM classifier on the SST-2 movie review sentiment classification task. Both models predict whether a sentence expresses positive or negative sentiment. The experiments examine classification performance, overfitting, sentence-length effects, training time, and attention patterns to understand how the two architectures behave on the same dataset.

## Dataset

- **Dataset:** Stanford Sentiment Treebank (SST-2), from GLUE
- **Task:** Binary sentiment classification
- **Classes:** Positive and negative
- **Vocabulary:** Built using the training split only
- **Data preparation:** Tokenization, numerical encoding, padding, and attention masks

The original training data was divided into training and validation subsets. The official GLUE validation set was used for final evaluation because the official test labels are hidden.

## Tech Stack

Python · PyTorch · Hugging Face Datasets · NumPy · Matplotlib · scikit-learn · Google Colab

## Workflow

1. Loaded SST-2 and created separate training and validation subsets while keeping the official validation set for final evaluation.
2. Built a vocabulary from training data only and implemented token encoding, batch padding, and attention masks.
3. Implemented scaled dot-product attention, multi-head attention, and a Transformer encoder block from scratch.
4. Built a Transformer classifier using token embeddings, positional encoding, one encoder block with four attention heads, masked mean pooling, and a classification layer.
5. Implemented an LSTM classifier as a baseline and used shared training and evaluation functions for both models.
6. Investigated Transformer overfitting by testing dropout, weight decay, minimum word frequency, training duration, and label smoothing.
7. Compared test accuracy, precision, recall, F1-score, confusion matrices, sentence-length performance, and training time.
8. Applied McNemar's test to compare model errors and visualized attention weights for correctly and incorrectly classified examples.

## Model Architecture

| Component | Transformer | LSTM |
|---|---|---|
| Input representation | Learned word embeddings | Learned word embeddings |
| Sequence processing | Self-attention with four heads | Recurrent LSTM layers |
| Positional information | Sinusoidal positional encoding | Sequential processing |
| Sentence representation | Masked mean pooling | LSTM-based representation |
| Output | Binary sentiment classifier | Binary sentiment classifier |

Both models were trained from scratch without pretrained language-model weights. Their parameter counts were similar, making the comparison useful for examining differences between the two architectures under the chosen setup.

## Results

### Overall Performance

| Metric | Transformer | LSTM |
|---|---:|---:|
| Test accuracy | 80.8% | 84.1% |
| Parameter count | ~2.00M | ~1.94M |
| Training time | ~90 seconds | ~130 seconds |

The LSTM achieved higher test accuracy, precision, and F1-score, while recall was approximately equal. The Transformer trained faster despite having a similar number of parameters.

### Error Analysis

The Transformer made more negative-to-positive classification errors than the LSTM (91 versus 68). Both models performed worse on longer sentences, but the LSTM outperformed the Transformer in every sentence-length group. The largest accuracy gap was 7.6 percentage points for sentences longer than 30 words, although this group contained only 105 examples.

McNemar's test gave a p-value of 0.018, indicating a statistically significant difference in the models' error patterns on this evaluation set.

### Overfitting and Regularization

The initial Transformer showed signs of overfitting as validation loss began rising while training loss continued to decrease. Experiments with dropout, weight decay, minimum word frequency, and training duration helped identify a more suitable configuration. Dropout of 0.3 and label smoothing were included in the final setup, with model settings selected using validation results.

### Attention Visualization

Attention heatmaps were generated for selected correct and incorrect predictions. The Transformer often assigned attention to sentiment-bearing words such as *charming*, *affecting*, and *slow*. However, some incorrect predictions still assigned attention to important sentiment words without producing the correct classification.

These examples illustrate that attention weights show where information is drawn from, but do not by themselves explain the model's final decision.

## Key Learnings

- **Self-attention:** Allows each token to gather information from other tokens in a sentence.
- **Multi-head attention:** Provides multiple attention patterns that can capture different relationships between words.
- **Padding masks:** Prevent padded positions from being treated as real words during attention and sentence pooling.
- **Overfitting:** Training loss alone is insufficient for selecting a model; validation performance is essential.
- **Architecture comparison:** A Transformer does not necessarily outperform an LSTM when trained from scratch on a limited dataset.
- **Error analysis:** Confusion matrices and sentence-length groups reveal weaknesses that overall accuracy can hide.
- **Statistical testing:** McNemar's test helps assess whether differences in paired model predictions are consistent with a meaningful performance gap.
- **Attention interpretation:** High attention to a sentiment-bearing word does not guarantee that the model will classify the sentence correctly.

## Limitations

- The Transformer uses only one encoder block and does not use pretrained representations.
- The vocabulary and word embeddings are learned from scratch, limiting the model's access to broader linguistic knowledge.
- The SST-2 training data contains many short phrases, while the official validation set contains more complete sentences. This distribution difference may contribute to the lower final evaluation scores.
- The custom validation split may contain related phrases from the same original sentence on both sides, potentially inflating validation performance.
- The official GLUE test set has hidden labels, so the official validation set was used for final evaluation. Consequently, the reported scores are not results on the hidden test set.
- The attention visualizations are qualitative examples and should not be treated as definitive explanations of model decisions.

## How to Run

1. Open the notebook in Google Colab and select a suitable runtime.
2. Run the setup cells to install or import the required libraries and configure the experiment.
3. Load SST-2, create the data splits, build the vocabulary, and prepare the DataLoaders.
4. Run the model implementation cells for attention, the Transformer encoder, and the LSTM baseline.
5. Train both models using the shared training and evaluation pipeline.
6. Run the evaluation cells to compare classification metrics, confusion matrices, sentence-length performance, and training time.
7. Generate attention heatmaps and review examples where the Transformer classified sentences correctly or incorrectly.

Training settings and other experiment parameters are managed through a central configuration, making it easier to adjust hyperparameters and repeat experiments.

## Conclusion

The experiments showed that the LSTM outperformed the single-block Transformer on SST-2, achieving 84.1% compared with 80.8% test accuracy, while requiring more training time. The Transformer displayed meaningful attention patterns but struggled more with negative-to-positive errors and longer sentences. Overall, this project demonstrates how to implement core Transformer components from scratch, compare architectures under a shared training pipeline, investigate overfitting, and use quantitative and qualitative evaluation to understand model behavior.
