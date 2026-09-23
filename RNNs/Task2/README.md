# GRU Stock Price Prediction & RNN vs. LSTM Vanishing Gradient Comparison

## Overview
This project implements a GRU in PyTorch to forecast next-day stock closing price, and separately runs a controlled empirical comparison between a vanilla RNN and an LSTM to demonstrate the vanishing gradient problem on long sequences.

## Dataset
Source: AAPL daily closing prices, 3-year window (`yfinance`)
Split: Chronological 70% train / 15% validation / 15% test (no shuffling, to respect time order)

## Tech Stack
Python · PyTorch · yfinance · scikit-learn (MinMaxScaler) · Matplotlib

## Workflow
- Initial approach predicted raw closing prices; this failed due to an extrapolation problem — AAPL's upward price trend meant the test period's price range fell outside what the model saw during training, producing flat, underestimated predictions (Test MAE $70.68)
- Fixed by reframing the prediction target as daily percentage returns instead of raw price levels, which stay small and bounded regardless of the underlying price trend
- GRU (64 hidden units) trained on 60-day return windows to predict next-day return
- Predicted returns reconstructed into prices using one-step-ahead evaluation — each day's predicted price anchored to the true previous day's actual price, avoiding compounding error
- Benchmarked against a naive "tomorrow = today" persistence baseline, and evaluated with Mean Directional Accuracy (MDA) to test genuine directional forecasting skill, not just magnitude closeness
- Separately, a vanilla RNN and an LSTM (identical hidden size, single layer, same data) were trained on long sequences (200 timesteps) of the same return data, with gradient magnitude measured at each individual timestep — for both the LSTM's hidden state (h_t) and cell state (c_t) — using a single shared training function, to empirically demonstrate vanishing gradients and isolate the architectural mechanism behind LSTM's advantage

## Model Architecture (GRU Forecasting)
| Layer | Type | Output Size |
|---|---|---|
| Input | 60-day return window | (batch, 60, 1) |
| GRU | 1 layer, 64 hidden units | (batch, 64) — final hidden state |
| Dropout | 0.2 | (batch, 64) |
| Output | Linear | (batch, 1) |

## Results

### GRU Stock Forecasting
| Approach | Test MAE | Test MSE |
|---|---|---|
| Raw price prediction | $70.68 | 5085.75 |
| Return-based prediction | **$3.93** | **31.36** |
| Naive baseline ("tomorrow = today") | $4.20 | 35.44 |

The return-based GRU modestly outperforms the naive baseline in magnitude terms (~6% lower MAE). However, **Mean Directional Accuracy (MDA) came out to 48.08%** — at or below random chance — indicating the model did not learn genuine directional forecasting skill (correctly predicting whether price moves up or down). This suggests the low MAE is driven largely by the strong influence of the true prior-day price anchor used in reconstruction, rather than real predictive ability, and reinforces the well-documented difficulty of short-term directional stock prediction. Magnitude-based metrics (MAE/MSE) alone can be misleading about a forecasting model's real-world usefulness — directional accuracy is a necessary complementary check.

### Vanishing Gradient Comparison (Vanilla RNN vs. LSTM)
Gradient magnitude was measured at each individual timestep's hidden state — and, for LSTM, cell state as well — via `retain_grad()` on a manually unrolled `RNNCell`/`LSTMCell` (in double precision, to avoid numerical underflow), using a single shared training function for both models. This is measured directly per timestep, rather than via the shared recurrent weight matrix, which sums gradient across all timesteps and isn't directly comparable between architectures with different gate counts.

| Distance from final timestep | Vanilla RNN (h_t) | LSTM (h_t) | LSTM (c_t) |
|---|---|---|---|
| 0 | 6.78e-03 | 7.73e-03 | 0.00* |
| 50 | 3.05e-13 | 2.12e-13 | 3.70e-13 |
| 100 | 1.11e-23 | 2.66e-23 | 5.33e-23 |
| 150 | 3.97e-34 | 3.62e-33 | 7.33e-33 |
| 199 | 2.39e-44 | 8.08e-43 | 1.61e-42 |

*The cell state gradient at distance 0 is structurally zero — the final `c_t` is never used downstream of the prediction layer in this implementation, not evidence of vanishing.

Gradient decays exponentially with distance across all three signals, confirming the vanishing gradient problem empirically. LSTM's hidden state gradient decays more slowly than vanilla RNN's (~34x larger at the furthest timestep), and LSTM's cell state — its additive, largely unfiltered gradient pathway — preserves signal even better than its own hidden state (~68x larger than vanilla RNN at the furthest timestep). This confirms that the cell state's additive update rule, not the hidden state alone, is the core architectural mechanism behind LSTM's resistance to vanishing gradients.

## Key Learnings
- Raw price-level forecasting on trending data is prone to extrapolation failure — neural networks generalize poorly outside their training range
- Reframing the target as returns keeps train/test distributions aligned and avoids extrapolation
- One-step-ahead evaluation (anchored to true prior values) is the correct reconstruction method; recursive (predicted-on-predicted) reconstruction compounds error
- Low MAE does not imply genuine predictive skill — Mean Directional Accuracy revealed the model's apparent accuracy was driven by the anchor effect rather than real directional forecasting ability
- Measuring gradient via the shared recurrent weight matrix is invalid evidence of vanishing gradients; per-timestep hidden-state (and cell-state) gradient tracking is the correct approach
- LSTM's hidden state (h_t) still passes through a saturating `tanh` each step and shares that weakness with vanilla RNN to a degree — the cell state (c_t) is the actual mechanism responsible for LSTM's advantage, and isolating it required manually unrolling the LSTM with `LSTMCell`
- PyTorch's fused `nn.RNN`/`nn.LSTM` kernels can mask true gradient decay even under double precision — manual unrolling was needed to observe the real, smooth decay curve

## How to Run
1. Clone this repository
2. Open the notebook in Jupyter Notebook or Google Colab
3. Run cells in order — AAPL data downloads automatically via `yfinance`

## Conclusion
Reframing stock forecasting as return prediction was essential to avoid extrapolation failure, but directional accuracy testing revealed the model's low error was largely an artifact of anchoring to true prior prices rather than genuine forecasting skill — a reminder that magnitude-based metrics alone can overstate a model's usefulness. The vanishing gradient experiment confirms, with direct per-timestep gradient measurements of both hidden and cell states, that LSTM's additive cell-state pathway — not just its gating in general — is the specific architectural mechanism that slows gradient decay relative to a vanilla RNN, motivating both LSTM's design and the eventual shift toward attention-based architectures like Transformers.
