import json

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and replace horizon_len = 6 to horizon_len = 60 in cell 4
cell_4_source = nb['cells'][4]['source']
for i, line in enumerate(cell_4_source):
    if "horizon_len = 6" in line:
        cell_4_source[i] = "horizon_len = 60 # 60 * 5m = 300 minutes (5 hours)\n"
        break

# In cell 0, update the text
nb['cells'][0]['source'][-1] = "We use the most recent 512 periods (approx 42 hours) to predict the next 60 periods (5 hours)."

# In cell 9, update visualization parameter vis_len to show more context, maybe 120
cell_9_source = nb['cells'][9]['source']
for i, line in enumerate(cell_9_source):
    if "vis_len = 60" in line:
        cell_9_source[i] = "vis_len = 120\n"
    if "Last 60 periods" in line:
        cell_9_source[i] = line.replace("Last 60 periods", "Last 120 periods")
    if "30 minutes (6 periods)" in line:
        cell_9_source[i] = line.replace("30 minutes (6 periods)", "5 hours (60 periods)")

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook horizon updated to 60 successfully.")
