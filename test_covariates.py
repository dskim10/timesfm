import numpy as np
from timesfm3.torch import TimesFM3Forecaster

forecaster = TimesFM3Forecaster.from_pretrained(
    "google/timesfm-3.0-pytorch",
    per_core_batch_size=4,
    device="cpu"
)

context = np.random.randn(512).astype(np.float32)
# shape testing: (covariate_len, features)
# wait, predict expects past_only_covariates of some shape.
# Let's try (512, 1) or (1, 512). Let's test if it accepts any.

try:
    poc = np.random.randn(512, 2).astype(np.float32)
    out = forecaster.predict(context=context, horizon=30, past_only_covariates=poc)
    print("Success with shape (512, 2)")
except Exception as e:
    print(f"Failed with shape (512, 2): {e}")

try:
    poc = np.random.randn(2, 512).astype(np.float32)
    out = forecaster.predict(context=context, horizon=30, past_only_covariates=poc)
    print("Success with shape (2, 512)")
except Exception as e:
    print(f"Failed with shape (2, 512): {e}")

