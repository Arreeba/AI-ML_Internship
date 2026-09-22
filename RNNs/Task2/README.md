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
- Benchmarked against a naive "tomorrow = today" persistence baseline
- Separately, a vanilla RNN and an LSTM (identical hidden size, single layer, same data) were trained on long sequences (200 timesteps) of the same return data, with gradient magnitude measured at each individual timestep's hidden state to empirically demonstrate vanishing gradients

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
| Return-based prediction | **$3.97** | **31.73** |
| Naive baseline ("tomorrow = today") | $4.20 | 35.44 |

The GRU's return-based predictions modestly outperform the naive baseline (~5.5% lower MAE), indicating genuine — if limited — predictive signal, consistent with the well-documented difficulty of short-term stock forecasting under near-random daily returns. A meaningful portion of the visual accuracy in the actual-vs-predicted plot reflects the strong influence of the true prior-day price anchor used in reconstruction, rather than the model alone forecasting large future moves — which is why the naive baseline comparison is the fairer measure of genuine model skill.

### Vanishing Gradient Comparison (Vanilla RNN vs. LSTM)
Gradient magnitude was measured at each individual timestep's hidden state (via `retain_grad()` on a manually unrolled `RNNCell`/`LSTMCell`, in double precision to avoid numerical underflow), rather than on the shared recurrent weight matrix (which sums gradient across all timesteps and isn't directly comparable between architectures with different gate counts).

| Distance from final timestep | Vanilla RNN gradient | LSTM gradient |
|---|---|---|
| 0 | 6.77e-03 | 7.72e-03 |
| 50 | 3.10e-13 | 2.12e-13 |
| 100 | 1.16e-23 | 2.65e-23 |
| 150 | 4.24e-34 | 3.61e-33 |
| 199 | 2.59e-44 | 8.05e-43 |

Both architectures show clear exponential gradient decay with distance, confirming the vanishing gradient problem empirically. LSTM's gradient decays more slowly than vanilla RNN's, with the gap widening at greater distances (~31x larger at the furthest timestep), demonstrating that LSTM's gating mechanism better preserves gradient signal across long sequences.

## Key Learnings
- Raw price-level forecasting on trending data is prone to extrapolation failure — neural networks generalize poorly outside their training range
- Reframing the target as returns (rather than price levels) is the standard fix and keeps train/test distributions aligned
- Recursive (predicted-on-predicted) reconstruction compounds error across the test period; one-step-ahead evaluation anchored to true prior values is the correct, standard evaluation method
- A model can look highly accurate on a price-reconstruction plot largely due to a strong ground-truth anchor effect — benchmarking against a naive baseline is essential to isolate genuine predictive skill
- Measuring gradient via the shared recurrent weight matrix is not valid evidence of vanishing gradients; per-timestep hidden-state gradient tracking is the correct approach
- PyTorch's fused `nn.RNN`/`nn.LSTM` kernels can mask true gradient decay even under double precision — manual unrolling via `RNNCell`/`LSTMCell` was needed to observe the real, smooth decay curve

## How to Run
1. Clone this repository
2. Open the notebook in Jupyter Notebook or Google Colab
3. Run cells in order — AAPL data downloads automatically via `yfinance`

## Conclusion
Reframing stock forecasting as return prediction (rather than raw price) was essential to avoid extrapolation failure, and the GRU shows modest but real predictive skill over a naive baseline once evaluated with correct one-step-ahead reconstruction. The vanishing gradient experiment confirms, with direct per-timestep gradient measurements, that LSTM's gating mechanism meaningfully slows gradient decay relative to a vanilla RNN — the core motivation for LSTM's design and, more broadly, for the shift toward attention-based architectures like Transformers.
