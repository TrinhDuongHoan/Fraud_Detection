import numpy as np
import pytest

from fraud_detection.config import Config
from fraud_detection.policy import ReviewState, business_metrics, daily_top_k, online_review
from fraud_detection.split import chronological_split


def test_splits_are_disjoint_days_and_chronological(transactions):
    parts = chronological_split(transactions, Config())
    previous = -1
    seen = set()
    for positions in parts.values():
        subset = transactions.iloc[positions]
        assert subset.TransactionDT.min() > previous
        assert not seen.intersection(positions)
        previous = subset.TransactionDT.max()
        seen.update(positions)
    assert len(seen) == len(transactions)


def test_policy_respects_every_prefix_and_batch_equivalence(transactions):
    scores = np.ones(len(transactions)) * 0.9
    all_at_once = online_review(transactions, scores, 0.5, 0.01)
    days = transactions.TransactionDT // 86400
    for day in days.unique():
        mask = days == day
        assert np.all(np.cumsum(all_at_once[mask]) <= np.floor(np.arange(1, mask.sum() + 1) * 0.01))
    state = ReviewState()
    cuts = [0, 77, 303, 1777, len(transactions)]
    batched = np.concatenate(
        [
            online_review(transactions.iloc[left:right], scores[left:right], 0.5, 0.01, state)
            for left, right in zip(cuts[:-1], cuts[1:], strict=True)
        ]
    )
    np.testing.assert_array_equal(all_at_once, batched)
    assert all_at_once.sum() == 30
    with pytest.raises(ValueError, match="chronological"):
        online_review(transactions.iloc[:10], scores[:10], 0.5, 0.01, state)


def test_future_scores_do_not_change_past_decisions(transactions):
    scores = np.full(len(transactions), 0.9)
    before = online_review(transactions, scores, 0.5, 0.01)
    scores[150:] = 0
    after = online_review(transactions, scores, 0.5, 0.01)
    np.testing.assert_array_equal(before[:150], after[:150])


def test_business_costs_and_daily_rounding(transactions):
    costs = business_metrics([1, 1, 0], [True, False, True], 100, 1, 0.8)
    assert costs["expected_savings"] == 78
    assert costs["review_precision"] == 0.5
    small = transactions.iloc[:99]
    assert not online_review(small, np.ones(99), 0.5, 0.01).any()
    assert not daily_top_k(small, np.ones(99), 0.01).any()
    assert daily_top_k(transactions, np.ones(len(transactions)), 0.01).sum() == 30


def test_sort_ties_are_deterministic(transactions):
    frame = transactions.iloc[:200].copy()
    frame["TransactionDT"] = 0
    shuffled = frame.sample(frac=1, random_state=4)
    selected = online_review(shuffled, np.ones(200), 0.5, 0.01)
    assert set(shuffled.loc[selected, "TransactionID"]) == {99, 199}


def test_threshold_one_disables_review(transactions):
    assert not online_review(transactions, np.ones(len(transactions)), 1.0, 0.01).any()


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(train_fraction=0.9)
    with pytest.raises(ValueError):
        Config(review_rate=0)
