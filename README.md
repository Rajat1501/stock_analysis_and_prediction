# Quantitative Stock Price Movement Prediction using Deep Learning

This project presents a sophisticated deep learning model designed to predict the next day's price movement (Up or Down) for a given stock, using historical time-series data enriched with various technical indicators. The model employs an efficient architecture combining Convolutional Neural Networks (CNNs), Bidirectional LSTMs, and an Attention mechanism to capture complex patterns in financial data.

## Table of Contents
1.  [Project Objective](#project-objective)
2.  [Technical Architecture](#technical-architecture)
3.  [Key Features](#key-features)
4.  [Dataset](#dataset)
5.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
6.  [Model Evaluation](#model-evaluation)
7.  [Future Work](#future-work)

---

### Project Objective

The primary goal is to build a robust binary classification model that provides a probabilistic forecast of whether a stock's closing price will be higher than the current day's close. This moves beyond simple price prediction to provide a more actionable trading signal, framed as a "buy" or "hold/sell" indicator. The project emphasizes a rigorous, production-ready approach, including a high-performance data pipeline, model checkpointing, and comprehensive evaluation.

---

### Technical Architecture

The model's architecture is specifically designed to interpret the complex, noisy, and sequential nature of financial market data.

1.  **Input Layer**: Accepts sequences of historical stock data, where each timestep contains multiple features (OHLCV + technical indicators).

2.  **1D Convolutional Layer (CNN)**: Acts as a feature detector. It scans for short-term, local patterns (e.g., a specific 3-day price action) across the sequence of technical indicators. This helps in extracting higher-level features from the raw data.

3.  **Bidirectional LSTM Layer (BiLSTM)**: This layer processes the sequences of features extracted by the CNN.
    * **LSTM**: Captures long-term temporal dependencies in the data.
    * **Bidirectional**: It processes the data in both forward (past to present) and backward (future to present) directions. In financial data, this allows the model to understand the context of a data point based on what happened before *and* after, leading to a more robust understanding of trends.

4.  **Attention Mechanism**: This is a critical layer. After the BiLSTM processes the entire sequence, the Attention layer allows the model to dynamically focus on the most influential timesteps when making its final prediction. For instance, it might learn that a sharp volume spike 10 days ago is more important than the stable price action from yesterday.

5.  **Output Layer**: A final Dense layer with a `sigmoid` activation function outputs a probability score between 0 and 1, representing the likelihood of the 'Target' class (price moving up).

---

### Key Features

* **High-Performance Data Pipeline**: Utilizes `tf.data.Dataset` for efficient, parallelized data loading and preprocessing, which is crucial for handling large financial datasets without memory bottlenecks.
* **Chronological Data Splitting**: Strictly maintains the temporal order of data in train, validation, and test sets to prevent lookahead bias, simulating a real-world trading scenario.
* **Feature Scaling**: Employs `MinMaxScaler` fitted *only* on the training data to prevent data leakage from the validation and test sets.
* **Advanced Callbacks**:
    * `ModelCheckpoint`: Saves only the best performing model on the validation set.
    * `EarlyStopping`: Prevents overfitting by halting training when validation performance degrades.
    * `ReduceLROnPlateau`: Adapts the learning rate for more stable convergence.
* **Comprehensive Evaluation**: The model isn't just evaluated on accuracy. It uses a suite of metrics (Precision, Recall, F1-Score, ROC AUC) to provide a holistic view of its performance, which is essential for imbalanced financial datasets.

---

### Dataset

The model is trained on 5-minute interval data for a stock from the NIFTY 100 index (e.g., DMART). The raw data (Open, High, Low, Close, Volume) is augmented with 14 key technical indicators:

* **Trend Indicators**: SMA (5, 20), EMA (5, 20), MACD
* **Momentum Indicators**: Momentum (10), RSI (14)
* **Volatility Indicators**: Bollinger Bands, ATR (Average True Range)
* **Oscillators**: Stochastic Oscillator (%K, %D), Williams %R

The **target variable** is a binary flag indicating if `Tomorrow's Close > Today's Close`.

---

### Getting Started

#### Prerequisites

* Python 3.8+
* TensorFlow 2.x
* Pandas
* NumPy
* Scikit-learn
* Joblib


#### Usage

1.  **Place your data**: Ensure CSV data file (e.g., `DMART_with_indicators_.csv`) is accessible. You can download the nifty50 5min data from kaggle      https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data
2.  **Configure the script**: Open the Python script and update the `FILEPATH` variable to point to your data file.
3.  **Run the training script**
    
The script will handle data preparation, model training, and evaluation. The best model will be saved as `best_stock_predictor.keras` and the data scaler as `data_scaler.joblib`.

---

### Model Evaluation

Upon completion of training, the script provides a detailed performance report on the unseen test data:

* **Accuracy**: Overall correctness.
* **Precision**: The accuracy of positive predictions. (How often is the "UP" prediction correct?)
* **Recall**: The model's ability to find all actual positive instances. (How many of the actual "UP" movements did the model catch?)
* **F1-Score**: The harmonic mean of Precision and Recall, providing a single score for model robustness.
* **ROC AUC**: The model's capability to distinguish between the positive and negative classes.

These metrics give a nuanced understanding of the model's real-world utility for financial forecasting.

---
--- Final Model Performance ---
* **Test Accuracy: 0.9061**
    -> What it means: Overall, what percentage of predictions were correct?

* **Test Precision: 0.9139**
    -> What it means: Of all the times the model predicted 'UP', how often was it right? (Measures signal quality)

* **Test Recall: 0.8841**
    -> What it means: Of all the times the market actually went 'UP', how many did the model catch? (Measures opportunity capture)

* **Test F1-Score: 0.8987**
    -> What it means: A balanced score between Precision and Recall. A good overall measure of a model's robustness.

* **Test ROC AUC: 0.9667**
    -> What it means: Measures the model's ability to distinguish between the 'UP' and 'DOWN' classes across all probability thresholds.

### Future Work

* **Hyperparameter Tuning**: Use KerasTuner or Optuna to systematically find the optimal set of hyperparameters (e.g., filter count, LSTM units, dropout rate).
* **Multi-Class Prediction**: Extend the model to predict "Strong Up", "Up", "Down", and "Strong Down" movements.
* **Feature Engineering**: Incorporate alternative data sources like market sentiment from news headlines or economic indicators.
* **Deployment**: Wrap the trained model in a REST API using Flask or FastAPI for real-time prediction inference.

