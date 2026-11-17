import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import merge_identity, normalize_columns, validate_transactions
from fraud_detection.features import FeatureBuilder


def test_identity_schema_and_left_join(transactions):
    identity = pd.DataFrame({"TransactionID": [1, 3], "id-01": [5.0, 7.0]})
    merged = merge_identity(transactions, identity)
    assert len(merged) == len(transactions)
    assert merged.TransactionID.tolist() == transactions.TransactionID.tolist()
    assert "id_01" in merged
    assert merged.id_01.notna().sum() == 2
    with pytest.raises(ValueError, match="unique"):
        merge_identity(transactions, pd.concat([identity, identity]))
    with pytest.raises(ValueError, match="collide"):
        normalize_columns(["id-01", "id_01"])


@pytest.mark.parametrize(
    "column,value",
    [
        ("TransactionAmt", -1),
        ("TransactionDT", np.inf),
        ("isFraud", 2),
    ],
)
def test_invalid_data_is_rejected(transactions, column, value):
    transactions[column] = transactions[column].astype(float)
    transactions.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_transactions(transactions, labeled=True)


def test_encoding_uses_only_training_and_ignores_target(transactions):
    train = transactions.iloc[:100].copy()
    train["card1"] = np.tile([1.0, 2.0], 50)
    builder = FeatureBuilder().fit(train)
    inference = transactions.iloc[100:110].drop(columns="isFraud").copy()
    inference["card1"] = 999.0
    inference["ProductCD"] = "NEW_CATEGORY"
    transformed = builder.transform(inference)
    assert transformed.card1.isna().all()
    assert transformed.ProductCD.isna().all()
    assert (transformed.card1_frequency == 0).all()
    assert not {"TransactionID", "isFraud", "TransactionDT"} & set(transformed.columns)
    changed = train.copy()
    changed["isFraud"] = 1 - changed.isFraud
    pd.testing.assert_frame_equal(builder.transform(train), builder.transform(changed))
    extra = train.assign(unrelated_column=np.nan)
    pd.testing.assert_frame_equal(builder.transform(train), builder.transform(extra))
    with pytest.raises(ValueError, match="Missing model input"):
        builder.transform(inference.drop(columns="TransactionAmt"))
