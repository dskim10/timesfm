import json

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Markdown Cell 0
nb['cells'][0]['source'] = [
    "# Bitcoin Price Prediction with TimesFM 3.0 (Advanced)\n",
    "\n",
    "This notebook demonstrates advanced techniques to improve forecasting for Bitcoin prices, including:\n",
    "1. **Multivariate Covariates**: Adding Open, High, Low, Trading Volume, SMA, and RSI (Relative Strength Index) to the model.\n",
    "2. **Context Length Optimization**: Increasing the context length to 512 for better trend capture.\n",
    "3. **5-Minute Intervals**: Reducing noise compared to 1-minute data.\n",
    "\n",
    "We use the most recent 512 periods (approx 42 hours) to predict the next 60 periods (5 hours)."
]

# Update cell 4 (Index 4) - we need to change horizon_len = 6 to horizon_len = 60
source4 = nb['cells'][4]['source']
for i, line in enumerate(source4):
    if line.startswith("horizon_len = 6"):
        source4[i] = "horizon_len = 60 # 60 * 5m = 5 hours\n"
nb['cells'][4]['source'] = source4

# Update cell 8 - prediction cell Markdown
# We should also change "the horizon (`6`)" to "the horizon (`60`)" if there is a markdown cell saying this.
source7 = nb['cells'][7]['source']
for i, line in enumerate(source7):
    if "the horizon (`6`)" in line:
        source7[i] = line.replace("the horizon (`6`)", "the horizon (`60`)")
    if "the horizon (`30`)" in line:
        source7[i] = line.replace("the horizon (`30`)", "the horizon (`60`)")
nb['cells'][7]['source'] = source7

# Update cell 9 (Index 9) - visualization markdown
source9 = nb['cells'][9]['source']
for i, line in enumerate(source9):
    if "30-minute ground truth" in line:
        source9[i] = line.replace("30-minute ground truth", "5-hour (60 periods) ground truth")
nb['cells'][9]['source'] = source9


# Update cell 10 (Index 10) - plotting code
source10 = nb['cells'][10]['source']
for i, line in enumerate(source10):
    if "vis_len = 60" in line:
        source10[i] = "vis_len = 120\n" # plot more context so it's visible next to 60 horizon
    if "30 minutes (6 periods)" in line:
        source10[i] = line.replace("30 minutes (6 periods)", "5 hours (60 periods)")
nb['cells'][10]['source'] = source10


with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully for horizon_len=60.")
