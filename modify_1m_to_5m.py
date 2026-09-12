import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bitcoin Price Prediction with TimesFM 3.0 (Advanced)\n",
    "\n",
    "This notebook demonstrates advanced techniques to improve forecasting for Bitcoin prices, including:\n",
    "1. **Multivariate Covariates**: Adding trading volume and moving averages to the model.\n",
    "2. **Context Length Optimization**: Increasing the context length to 512 for better trend capture.\n",
    "3. **5-Minute Intervals**: Reducing noise compared to 1-minute data.\n",
    "\n",
    "We use the most recent 512 periods (approx 42 hours) to predict the next 6 periods (30 minutes)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from timesfm3.torch import TimesFM3Forecaster\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Fetch Data (5m Interval) & Feature Engineering\n",
    "\n",
    "We fetch 60 days of 5-minute data. We will also compute a Simple Moving Average (SMA) and include Volume as our `past_only_covariates`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "ticker = \"BTC-USD\"\n",
    "data = yf.download(ticker, period=\"60d\", interval=\"5m\")\n",
    "\n",
    "if isinstance(data.columns, pd.MultiIndex):\n",
    "    close_prices = data['Close'][ticker]\n",
    "    volume = data['Volume'][ticker]\n",
    "else:\n",
    "    close_prices = data['Close']\n",
    "    volume = data['Volume']\n",
    "\n",
    "# Feature Engineering: Moving Average\n",
    "sma_window = 12 # 1 hour SMA\n",
    "sma = close_prices.rolling(window=sma_window).mean()\n",
    "\n",
    "# Drop NA from rolling window\n",
    "valid_idx = sma.dropna().index\n",
    "close_prices = close_prices.loc[valid_idx].values\n",
    "volume = volume.loc[valid_idx].values\n",
    "sma = sma.loc[valid_idx].values\n",
    "dates = valid_idx\n",
    "\n",
    "# Set up the evaluation slice\n",
    "context_len = 512\n",
    "horizon_len = 6 # 6 * 5m = 30 minutes\n",
    "eval_len = context_len + horizon_len\n",
    "\n",
    "recent_prices = close_prices[-eval_len:].astype(np.float32)\n",
    "recent_volume = volume[-eval_len:].astype(np.float32)\n",
    "recent_sma = sma[-eval_len:].astype(np.float32)\n",
    "recent_dates = dates[-eval_len:]\n",
    "\n",
    "# Split into context (past) and ground truth (future to predict)\n",
    "context_prices = recent_prices[:context_len]\n",
    "context_volume = recent_volume[:context_len]\n",
    "context_sma = recent_sma[:context_len]\n",
    "ground_truth = recent_prices[context_len:]\n",
    "\n",
    "context_dates = recent_dates[:context_len]\n",
    "gt_dates = recent_dates[context_len:]\n",
    "\n",
    "# Normalize covariates (Important for stable inference when using covariates)\n",
    "scaler_vol = StandardScaler()\n",
    "scaler_sma = StandardScaler()\n",
    "\n",
    "norm_volume = scaler_vol.fit_transform(context_volume.reshape(-1, 1)).flatten()\n",
    "norm_sma = scaler_sma.fit_transform(context_sma.reshape(-1, 1)).flatten()\n",
    "\n",
    "# TimesFM expects past_only_covariates in shape (num_features, context_len)\n",
    "past_only_covs = np.stack([norm_volume, norm_sma], axis=0)\n",
    "\n",
    "print(f\"Context length: {len(context_prices)} periods (5m)\")\n",
    "print(f\"Ground truth length: {len(ground_truth)} periods (5m)\")\n",
    "print(f\"Covariates shape: {past_only_covs.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load TimesFM 3.0 Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecaster = TimesFM3Forecaster.from_pretrained(\n",
    "    \"google/timesfm-3.0-pytorch\",\n",
    "    per_core_batch_size=4,\n",
    "    device=\"cpu\"\n",
    ")\n",
    "print(f\"TimesFM3 loaded on device: {forecaster.device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Multivariate Inference\n",
    "We provide the target series (`context_prices`), the horizon (`6`), and our engineered `past_only_covs`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch_out = forecaster.predict(\n",
    "    context=context_prices,\n",
    "    horizon=horizon_len,\n",
    "    past_only_covariates=past_only_covs,\n",
    "    return_quantiles=True,\n",
    ")\n",
    "\n",
    "print(\"Forecast generated successfully!\")\n",
    "print(f\"Point forecast shape: {torch_out.forecast.shape}\")\n",
    "\n",
    "# Calculate Error (MAE)\n",
    "mae = np.mean(np.abs(torch_out.forecast - ground_truth))\n",
    "print(f\"Mean Absolute Error (MAE): {mae:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualize Results\n",
    "Plot the zoomed-in context alongside the actual 30-minute ground truth and the model's forecast."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 7))\n",
    "\n",
    "# For visualization, only show the last 60 context points to clearly see the horizon\n",
    "vis_len = 60\n",
    "\n",
    "plt.plot(context_dates[-vis_len:], context_prices[-vis_len:], label=\"Context (Last 60 periods)\", color=\"black\", linewidth=1.5)\n",
    "\n",
    "# Plot the 30 minutes (6 periods) of actual ground truth\n",
    "plt.plot(gt_dates, ground_truth, label=\"Actual BTC Price (Ground Truth)\", color=\"gray\", linestyle=\"--\", linewidth=2.0)\n",
    "\n",
    "# Plot the forecast median (point forecast)\n",
    "plt.plot(gt_dates, torch_out.forecast, label=\"TimesFM3 Forecast (p50)\", color=\"#ff7f0e\", linewidth=2.5)\n",
    "\n",
    "# Plot uncertainty intervals\n",
    "if torch_out.quantiles is not None:\n",
    "    plt.fill_between(\n",
    "        gt_dates,\n",
    "        torch_out.quantiles[:, 0],   # usually p10\n",
    "        torch_out.quantiles[:, -1],  # usually p90\n",
    "        color=\"salmon\",\n",
    "        alpha=0.3,\n",
    "        label=\"p10-p90 Prediction Interval\",\n",
    "    )\n",
    "\n",
    "plt.title(f\"Bitcoin Price Advanced Forecast (5-Minute Interval)\\nHorizon = {horizon_len} periods, Context = {context_len} periods\", fontsize=14)\n",
    "plt.ylabel(\"Price (USD)\", fontsize=12)\n",
    "plt.xlabel(\"Time\", fontsize=12)\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Advanced notebook created.")
