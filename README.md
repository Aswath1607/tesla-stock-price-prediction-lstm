# Tesla Stock Price Forecasting Using LSTM

## Open the project in Google Colab

**[Open the Tesla LSTM notebook in Google Colab](https://colab.research.google.com/drive/1lQOmfmM_PKozf2d-nnflKKYcr-rnrzIJ?usp=sharing)**

Make sure the notebook is shared with **Anyone with the link — Viewer** if your evaluator needs to open it.

## Project Overview
This project applies a Long Short-Term Memory (LSTM) neural network to historical Tesla stock prices. The notebook uses the previous 60 observations of Tesla's opening price to predict the next observation's opening price.

> Educational machine-learning project only. This model is not financial advice.

## Files
- `tesla_stock_price_prediction_lstm.ipynb` — notebook containing data exploration, preprocessing, LSTM training, evaluation, and plots.
- `Tesla.csv` — historical Tesla stock dataset used by the notebook.
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignores local environments and generated model files.

## Dataset
The CSV should contain a `Date` column and an `Open` column. The notebook uses `Open` as the target series. Keep attribution to the original dataset source if required by your course.

## Preprocessing
1. Parse dates and set `Date` as the index.
2. Sort observations chronologically.
3. Select the `Open` price and remove missing target rows.
4. Scale values to `[0, 1]` using `MinMaxScaler`.
5. Split chronologically: first 75% for training and final 25% for testing.
6. Build sliding windows of 60 previous observations to forecast the next value.

## LSTM Architecture
- LSTM: 50 units, `return_sequences=True`
- LSTM: 64 units
- Dense: 32 units
- Dense: 16 units
- Dense output: 1 unit

The model uses Adam optimization and mean squared error loss. Early stopping is used during training.

## Evaluation and Results
The notebook calculates:
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error

It also plots training prices, actual test prices, and model predictions. Run the notebook to produce the metric values. Add the measured RMSE/MAE and your observations from the graph here after a successful run; do not invent results.

## Run in Google Colab
1. Open the [Colab notebook](https://colab.research.google.com/drive/1lQOmfmM_PKozf2d-nnflKKYcr-rnrzIJ?usp=sharing).
2. Confirm that `Tesla.csv` is available to the runtime. If prompted, upload the CSV from this repository.
3. Run the cells in order or select **Runtime → Run all**.
4. Review the RMSE, MAE, and prediction plot.
5. The notebook saves the trained model as `tesla_stock_model.keras` in the runtime.

## Run Locally
Use a Python environment compatible with TensorFlow, then run:

```bash
pip install -r requirements.txt
jupyter notebook tesla_stock_price_prediction_lstm.ipynb
```

## Limitations and Improvements
This version uses only historical opening prices. Future improvements could include other features such as High, Low, Close, and Volume; a chronological validation set; a naive baseline; and evaluation across multiple forecast windows.
