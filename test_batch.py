import numpy as np
import time
from timesfm3.torch import TimesFM3Forecaster

forecaster = TimesFM3Forecaster.from_pretrained(
    "google/timesfm-3.0-pytorch",
    per_core_batch_size=32,
    device="cpu"
)

num_windows = 100
contexts = [np.random.randn(512).astype(np.float32) for _ in range(num_windows)]

t0 = time.time()
results = list(forecaster.predict_batch(
    contexts=contexts,
    horizon=1,
))
t1 = time.time()

print(f"Time for {num_windows} predictions: {t1-t0:.2f} seconds")
