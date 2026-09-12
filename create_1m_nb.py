import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bitcoin Price Prediction with TimesFM 3.0 (1-Minute Intervals)\n",
    "\n",
    "This notebook demonstrates how to use the Google TimesFM 3.0 model to forecast Bitcoin prices on a **1-minute granularity**. We will use the most recent 120 minutes of price data as context to predict the next 30 minutes, and evaluate the model by comparing the predictions to the actual ground truth."
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
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Fetch Bitcoin Data (1m Interval)\n",
    "\n",
    "Note: Yahoo Finance API allows a maximum of 7 days for 1-minute granularity data. We will fetch the last 7 days and extract the final 150 minutes (120 min for context, 30 min for ground truth evaluation)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "ticker = \"BTC-USD\"\n",
    "data = yf.download(ticker, period=\"7d\", interval=\"1m\")\n",
    "\n",
    "# Handle potential MultiIndex columns from newer yfinance versions\n",
    "if isinstance(data.columns, pd.MultiIndex):\n",
    "    close_prices = data['Close'][ticker].values\n",
    "    dates = data.index\n",
    "else:\n",
    "    close_prices = data['Close'].values\n",
    "    dates = data.index\n",
    "\n",
    "# Set up the evaluation slice\n",
    "eval_len = 150\n",
    "horizon_len = 30\n",
    "context_len = eval_len - horizon_len\n",
    "\n",
    "recent_prices = close_prices[-eval_len:].astype(np.float32)\n",
    "recent_dates = dates[-eval_len:]\n",
    "\n",
    "# Split into context (past) and ground truth (future to predict)\n",
    "context = recent_prices[:context_len]\n",
    "ground_truth = recent_prices[context_len:]\n",
    "context_dates = recent_dates[:context_len]\n",
    "gt_dates = recent_dates[context_len:]\n",
    "\n",
    "print(f\"Fetched total data points: {len(close_prices)}\")\n",
    "print(f\"Context length: {len(context)} minutes\")\n",
    "print(f\"Ground truth length: {len(ground_truth)} minutes\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load TimesFM 3.0 Model\n",
    "We use `device=\"cpu\"` to ensure maximum compatibility."
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
    "## 4. Run Zero-Shot Inference\n",
    "We provide the context to the forecaster and set `horizon=30` to predict the next 30 minutes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch_out = forecaster.predict(\n",
    "    context=context,\n",
    "    horizon=horizon_len,\n",
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
    "Plot the 120-minute context alongside the actual 30-minute ground truth and the model's forecast."
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
    "# Plot the 120 minutes of context\n",
    "plt.plot(context_dates, context, label=\"Context (Input 120m)\", color=\"black\", linewidth=1.5)\n",
    "\n",
    "# Plot the 30 minutes of actual ground truth\n",
    "plt.plot(gt_dates, ground_truth, label=\"Actual BTC Price (Ground Truth 30m)\", color=\"gray\", linestyle=\"--\", linewidth=2.0)\n",
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
    "plt.title(f\"Bitcoin Price Forecast (1-Minute Interval)\\nHorizon = {horizon_len} min, Context = {context_len} min\", fontsize=14)\n",
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

with open("timesfm3-usage/notebooks/bitcoin_prediction_1m_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
