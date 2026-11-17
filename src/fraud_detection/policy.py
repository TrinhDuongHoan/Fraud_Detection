from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.data import ID, TIME


@dataclass
class ReviewState:
    day: int | None = None
    seen: int = 0
    reviewed: int = 0
    last_time: float = -1.0
    last_id: int = -1


def online_review(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    rate: float,
    state: ReviewState | None = None,
) -> np.ndarray:
    if not 0 < rate <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid policy rate or threshold")
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("Scores must be finite and aligned with input")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Scores must lie in [0, 1]")
    state = state if state is not None else ReviewState()
    times = frame[TIME].to_numpy()
    ids = frame[ID].to_numpy()
    order = np.lexsort((ids, times))
    if len(order) and (times[order[0]], ids[order[0]]) <= (state.last_time, state.last_id):
        raise ValueError("Policy batches must arrive in strictly chronological (time, ID) order")
    if len(set(zip(times, ids, strict=True))) != len(frame):
        raise ValueError("Duplicate policy events")
    decisions = np.zeros(len(frame), dtype=bool)
    for index in order:
        day = int(times[index] // 86400)
        if day != state.day:
            state.day, state.seen, state.reviewed = day, 0, 0
        state.seen += 1
        capacity = int(np.floor(rate * state.seen + 1e-12))
        if probabilities[index] > threshold and state.reviewed < capacity:
            decisions[index] = True
            state.reviewed += 1
        state.last_time, state.last_id = float(times[index]), int(ids[index])
    return decisions


def daily_top_k(frame: pd.DataFrame, scores: np.ndarray, rate: float) -> np.ndarray:
    ranked = pd.DataFrame(
        {
            "day": (frame[TIME] // 86400).to_numpy(),
            "score": scores,
            "id": frame[ID].to_numpy(),
            "position": np.arange(len(frame)),
        }
    ).sort_values(["day", "score", "id"], ascending=[True, False, True])
    capacity = np.floor(ranked.groupby("day")["id"].transform("size") * rate + 1e-12)
    selected = ranked.groupby("day").cumcount() < capacity
    result = np.zeros(len(frame), dtype=bool)
    result[ranked.loc[selected, "position"].to_numpy()] = True
    return result


def business_metrics(
    labels: np.ndarray,
    selected: np.ndarray,
    fraud_loss: float,
    review_cost: float,
    sensitivity: float,
) -> dict:
    labels, selected = np.asarray(labels), np.asarray(selected, dtype=bool)
    fraud = int(labels.sum())
    reviewed = int(selected.sum())
    captured = int(labels[selected].sum())
    baseline_cost = fraud * fraud_loss
    prevented = captured * sensitivity * fraud_loss
    expected_cost = baseline_cost - prevented + reviewed * review_cost
    return {
        "reviews": reviewed,
        "review_rate": reviewed / len(labels) if len(labels) else 0.0,
        "fraud_in_review": captured,
        "review_precision": captured / reviewed if reviewed else 0.0,
        "review_recall": captured / fraud if fraud else 0.0,
        "expected_prevented_frauds": captured * sensitivity,
        "no_review_cost": baseline_cost,
        "expected_cost": expected_cost,
        "expected_savings": baseline_cost - expected_cost,
    }


def select_threshold(frame: pd.DataFrame, scores: np.ndarray, config) -> tuple[float, pd.DataFrame]:
    minimum = (
        config.review_cost / (config.review_sensitivity * config.fraud_loss)
        if config.review_sensitivity
        else 1.0
    )
    candidates = np.unique(
        np.r_[
            min(minimum, 1.0),
            np.quantile(scores, np.linspace(0.50, 1.0, 41)),
            1.0,
        ]
    )
    candidates = candidates[candidates >= min(minimum, 1.0)]
    rows = []
    for threshold in candidates:
        selected = online_review(frame, scores, float(threshold), config.review_rate)
        rows.append(
            {
                "threshold": float(threshold),
                **business_metrics(
                    frame["isFraud"].to_numpy(),
                    selected,
                    config.fraud_loss,
                    config.review_cost,
                    config.review_sensitivity,
                ),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(
            ["expected_savings", "threshold"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    return float(table.iloc[0]["threshold"]), table
