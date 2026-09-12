import json

with open('timesfm3-usage/notebooks/bitcoin_prediction_timesfm3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fetch_code = """ticker = "BTC-USD"
data = yf.download(ticker, period="2y", interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close'][ticker].values
    dates = data.index
else:
    close_prices = data['Close'].values
    dates = data.index

# We want the last 150 days total for this evaluation
# 120 days for context, 30 days for ground truth
eval_len = 150
horizon_len = 30
context_len = eval_len - horizon_len

recent_prices = close_prices[-eval_len:].astype(np.float32)
recent_dates = dates[-eval_len:]

# Split into context and ground truth
context = recent_prices[:context_len]
ground_truth = recent_prices[context_len:]
context_dates = recent_dates[:context_len]
gt_dates = recent_dates[context_len:]

print(f"Context length: {len(context)} days")
print(f"Ground truth length: {len(ground_truth)} days")"""
nb['cells'][3]['source'] = [line + "\n" for line in fetch_code.split('\n')]
nb['cells'][3]['source'][-1] = nb['cells'][3]['source'][-1].rstrip('\n')

pred_code = """# Run zero-shot forecast on the context to predict the next 30 days
torch_out = forecaster.predict(
    context=context,
    horizon=horizon_len,
    return_quantiles=True,
)

print("Forecast generated successfully!")
print(f"Point forecast shape: {torch_out.forecast.shape}")

# Calculate Error (MAE)
mae = np.mean(np.abs(torch_out.forecast - ground_truth))
print(f"Mean Absolute Error (MAE): {mae:.2f}")"""
nb['cells'][7]['source'] = [line + "\n" for line in pred_code.split('\n')]
nb['cells'][7]['source'][-1] = nb['cells'][7]['source'][-1].rstrip('\n')

plot_code = """plt.figure(figsize=(14, 7))

# Plot the 120 days of context
plt.plot(context_dates, context, label="Context (Input)", color="black", linewidth=1.5)

# Plot the 30 days of actual ground truth
plt.plot(gt_dates, ground_truth, label="Actual BTC Price (Ground Truth)", color="gray", linestyle="--", linewidth=2.0)

# Plot the forecast median (point forecast)
plt.plot(gt_dates, torch_out.forecast, label="TimesFM3 Forecast (p50)", color="#ff7f0e", linewidth=2.5)

# Plot uncertainty intervals
if torch_out.quantiles is not None:
    plt.fill_between(
        gt_dates,
        torch_out.quantiles[:, 0],   # usually p10
        torch_out.quantiles[:, -1],  # usually p90
        color="salmon",
        alpha=0.3,
        label="p10-p90 Prediction Interval",
    )

plt.title(f"Bitcoin Price Forecast Evaluation (Horizon = {horizon_len} days, Context = {context_len} days)", fontsize=14)
plt.ylabel("Price (USD)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
nb['cells'][9]['source'] = [line + "\n" for line in plot_code.split('\n')]
nb['cells'][9]['source'][-1] = nb['cells'][9]['source'][-1].rstrip('\n')

with open('timesfm3-usage/notebooks/bitcoin_prediction_timesfm3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
