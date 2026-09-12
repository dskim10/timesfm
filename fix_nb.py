import json

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "valid_idx = sma.dropna().intersection" in line:
                line = line.replace("sma.dropna().intersection", "sma.dropna().index.intersection")
            new_source.append(line)
        cell['source'] = new_source

with open("timesfm3-usage/notebooks/bitcoin_prediction_advanced_timesfm3.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
