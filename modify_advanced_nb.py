import json

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 4 (Index 4) with RSI, Open, High, Low
new_feature_code = """ticker = "BTC-USD"
data = yf.download(ticker, period="60d", interval="5m")

if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close'][ticker]
    open_prices = data['Open'][ticker]
    high_prices = data['High'][ticker]
    low_prices = data['Low'][ticker]
    volume = data['Volume'][ticker]
else:
    close_prices = data['Close']
    open_prices = data['Open']
    high_prices = data['High']
    low_prices = data['Low']
    volume = data['Volume']

# Feature Engineering: Moving Average (1 hour SMA)
sma_window = 12 
sma = close_prices.rolling(window=sma_window).mean()

# Feature Engineering: RSI (14 period)
delta = close_prices.diff()
gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Drop NA from rolling windows and indicators
valid_idx = sma.dropna().intersection(rsi.dropna().index)

close_prices = close_prices.loc[valid_idx].values
open_prices = open_prices.loc[valid_idx].values
high_prices = high_prices.loc[valid_idx].values
low_prices = low_prices.loc[valid_idx].values
volume = volume.loc[valid_idx].values
sma = sma.loc[valid_idx].values
rsi = rsi.loc[valid_idx].values
dates = valid_idx

# Set up the evaluation slice
context_len = 512
horizon_len = 6 # 6 * 5m = 30 minutes
eval_len = context_len + horizon_len

recent_prices = close_prices[-eval_len:].astype(np.float32)
recent_open = open_prices[-eval_len:].astype(np.float32)
recent_high = high_prices[-eval_len:].astype(np.float32)
recent_low = low_prices[-eval_len:].astype(np.float32)
recent_volume = volume[-eval_len:].astype(np.float32)
recent_sma = sma[-eval_len:].astype(np.float32)
recent_rsi = rsi[-eval_len:].astype(np.float32)
recent_dates = dates[-eval_len:]

# Split into context (past) and ground truth (future to predict)
context_prices = recent_prices[:context_len]
context_open = recent_open[:context_len]
context_high = recent_high[:context_len]
context_low = recent_low[:context_len]
context_volume = recent_volume[:context_len]
context_sma = recent_sma[:context_len]
context_rsi = recent_rsi[:context_len]
ground_truth = recent_prices[context_len:]

context_dates = recent_dates[:context_len]
gt_dates = recent_dates[context_len:]

# Normalize covariates (Important for stable inference when using covariates)
scaler_open = StandardScaler()
scaler_high = StandardScaler()
scaler_low = StandardScaler()
scaler_vol = StandardScaler()
scaler_sma = StandardScaler()
scaler_rsi = StandardScaler()

norm_open = scaler_open.fit_transform(context_open.reshape(-1, 1)).flatten()
norm_high = scaler_high.fit_transform(context_high.reshape(-1, 1)).flatten()
norm_low = scaler_low.fit_transform(context_low.reshape(-1, 1)).flatten()
norm_volume = scaler_vol.fit_transform(context_volume.reshape(-1, 1)).flatten()
norm_sma = scaler_sma.fit_transform(context_sma.reshape(-1, 1)).flatten()
norm_rsi = scaler_rsi.fit_transform(context_rsi.reshape(-1, 1)).flatten()

# TimesFM expects past_only_covariates in shape (num_features, context_len)
past_only_covs = np.stack([
    norm_open, 
    norm_high, 
    norm_low, 
    norm_volume, 
    norm_sma, 
    norm_rsi
], axis=0)

print(f"Context length: {len(context_prices)} periods (5m)")
print(f"Ground truth length: {len(ground_truth)} periods (5m)")
print(f"Covariates shape: {past_only_covs.shape}")"""

nb['cells'][4]['source'] = [line + "\n" for line in new_feature_code.split('\n')]
nb['cells'][4]['source'][-1] = nb['cells'][4]['source'][-1].rstrip('\n')

# Update cell 0 (Markdown) to mention new features
nb['cells'][0]['source'] = [
    "# Bitcoin Price Prediction with TimesFM 3.0 (Advanced)\n",
    "\n",
    "This notebook demonstrates advanced techniques to improve forecasting for Bitcoin prices, including:\n",
    "1. **Multivariate Covariates**: Adding Open, High, Low, Trading Volume, SMA, and RSI (Relative Strength Index) to the model.\n",
    "2. **Context Length Optimization**: Increasing the context length to 512 for better trend capture.\n",
    "3. **5-Minute Intervals**: Reducing noise compared to 1-minute data.\n",
    "\n",
    "We use the most recent 512 periods (approx 42 hours) to predict the next 6 periods (30 minutes)."
]

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with new covariates.")
