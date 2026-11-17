import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def transactions():
    rng = np.random.default_rng(42)
    size = 3000
    return pd.DataFrame(
        {
            "TransactionID": np.arange(size),
            "TransactionDT": np.repeat(np.arange(30), 100) * 86400 + np.tile(np.arange(100), 30),
            "TransactionAmt": rng.lognormal(4, 1, size).astype("float32"),
            "ProductCD": pd.Series(rng.choice(["W", "C", "H"], size), dtype="string"),
            "card1": rng.integers(1, 30, size).astype("float32"),
            "isFraud": (rng.random(size) < 0.12).astype("float32"),
        }
    )
