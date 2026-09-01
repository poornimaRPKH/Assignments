# Assignment 10 — RNN & LSTM Time Series Prediction

## 📌 Project Overview
This project implements and compares two recurrent neural network architectures —
a **Simple RNN** and an **LSTM (Long Short-Term Memory)** network — for univariate
time series forecasting. The goal is to predict future values of a sequential dataset
given a window of past observations, and to evaluate which architecture generalizes
better on unseen (test) data.

The full workflow — data generation/preprocessing, sequence construction, model
building, training, prediction, visualization, and evaluation — is implemented in a
single, self-contained Jupyter/Colab notebook.

## 🎯 Objectives
- Preprocess a time series for supervised deep-learning (scaling + windowing).
- Build and train a Simple RNN model.
- Build and train an LSTM model.
- Compare predictions of both models against the actual values.
- Evaluate and compare performance using **RMSE** and **MAE**.

## 🗂️ Repository Structure
```
.
├── Assignment10_RNN_LSTM_TimeSeries.ipynb   # Main notebook (code + outputs)
├── README.md                                # Project documentation (this file)
├── requirements.txt                         # Python dependencies
├── data/
│   ├── synthetic_timeseries.csv             # Generated input dataset
│   └── evaluation_metrics.csv               # Final RMSE / MAE results
└── images/
    ├── raw_series.png                       # Raw time series plot
    ├── rnn_training_curve.png               # RNN train/val loss curve
    ├── lstm_training_curve.png              # LSTM train/val loss curve
    ├── prediction_comparison.png            # Actual vs RNN vs LSTM (full test set)
    ├── prediction_comparison_zoom.png       # Zoomed-in comparison (first 100 points)
    └── metrics_comparison.png               # RMSE / MAE bar chart comparison
```

## 📊 Dataset
The notebook generates a **synthetic univariate time series** of 1,000 daily points
composed of:
- a linear **trend** component,
- an **annual seasonal** sine component,
- a shorter **weekly seasonal** ripple,
- and **Gaussian noise**.

This keeps the notebook fully reproducible and independent of external downloads,
while still exhibiting the trend + seasonality + noise structure typical of real-world
series (sales, temperature, traffic, etc.). The generated data is saved to
`data/synthetic_timeseries.csv` — you can swap in any real univariate CSV (e.g. Airline
Passengers, stock closing prices, weather data) by replacing this file and adjusting
the load step.

## ⚙️ Methodology

### 1. Preprocessing
- Scale the series to `[0, 1]` using `MinMaxScaler`.
- Split chronologically into train (80%) / test (20%) — **no shuffling**, to preserve
  temporal order.

### 2. Sequence Generation
- Convert the scaled series into supervised sequences using a sliding window of
  `LOOKBACK = 30` past time steps to predict the next value.
- Reshape into the `[samples, timesteps, features]` format required by Keras
  recurrent layers.

### 3. Models
Both models share the same overall depth/capacity so the comparison isolates the
effect of the recurrent cell type:

| Layer | Simple RNN Model | LSTM Model |
|---|---|---|
| 1 | `SimpleRNN(64, return_sequences=True)` | `LSTM(64, return_sequences=True)` |
| 2 | `Dropout(0.2)` | `Dropout(0.2)` |
| 3 | `SimpleRNN(32)` | `LSTM(32)` |
| 4 | `Dropout(0.2)` | `Dropout(0.2)` |
| 5 | `Dense(1)` | `Dense(1)` |

- Optimizer: `adam`, Loss: `mse`, Metric: `mae`
- Trained for up to 60 epochs with `EarlyStopping` (patience=8, restores best weights).

### 4. Evaluation
Predictions are inverse-transformed back to the original value scale, then compared
against actual test values using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

## 📈 Results

| Model | RMSE | MAE |
|---|---|---|
| Simple RNN | ~2.26 | ~1.81 |
| LSTM | ~2.77 | ~2.35 |

*(Exact values are written to `data/evaluation_metrics.csv` on each run and may vary
slightly run-to-run.)*

In this run, the Simple RNN slightly outperformed the LSTM: it tracked the
short-period weekly ripple in the data more closely, while the LSTM produced a
smoother, more trend-following forecast. See the notebook's Conclusion section for a
full discussion of why this can happen and how it depends on look-back window size,
training duration, and the nature of the underlying signal.

## 🖼️ Sample Outputs
See the `images/` folder for:
- The raw input series
- Training/validation loss curves for both models
- Actual vs. predicted comparison plots (full test set and zoomed-in view)
- A bar chart comparing RMSE/MAE across both models

## 🚀 How to Run

### Option A — Google Colab
1. Upload `Assignment10_RNN_LSTM_TimeSeries.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run all cells (`Runtime > Run all`). No external dataset download is required.

### Option B — Local Jupyter
```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Assignment10_RNN_LSTM_TimeSeries.ipynb
```

## 🧰 Requirements
See `requirements.txt`:
```
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

## 🔮 Possible Extensions
- Try GRU cells or stacked/bidirectional LSTMs.
- Experiment with different `LOOKBACK` window sizes.
- Use multivariate inputs (exogenous regressors).
- Perform a systematic hyperparameter search (units, dropout rate, learning rate).
- Swap in a real-world dataset (stock prices, energy demand, weather, etc.).

## 📄 License
This project is provided for educational purposes as part of a coursework assignment.
