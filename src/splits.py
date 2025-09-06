from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


def _as_ts(x): return pd.to_datetime(x).tz_localize(None)


def build_rolling_folds(
    dts: pd.Series,
    eligible_mask: np.ndarray,
    folds: int,
    val_len_days: int,
    purge_days: int
) -> List[Dict]:
    """
    Create forward-chaining folds on the eligible time span.
    Each fold i uses:
      - VAL: [val_start_i, val_end_i]
      - TRAIN: <= (val_start_i - purge_days)
    Returns list of dicts: {"val_start","val_end","train_mask","val_mask"}
    """
    dts = pd.to_datetime(dts)
    rest_idx = np.where(eligible_mask)[0]
    if rest_idx.size == 0:
        return []

    rest_dates = dts[rest_idx]
    start_rest = rest_dates.min()
    end_rest = rest_dates.max()

    total_days = (end_rest - start_rest).days
    if total_days <= val_len_days + purge_days + 5:
        # too short; fallback to single fold at the end
        val_start = end_rest - pd.Timedelta(days=val_len_days)
        val_end = end_rest
        val_mask = (eligible_mask & (dts >= val_start)
                    & (dts <= val_end)).values
        train_mask = (eligible_mask & (
            dts < (val_start - pd.Timedelta(days=purge_days)))).values
        return [{"val_start": val_start, "val_end": val_end, "train_mask": train_mask, "val_mask": val_mask}]

    # space fold starts across the interval
    usable_days = total_days - val_len_days
    step_days = max(1, usable_days // max(1, folds))
    fold_starts = [start_rest +
                   pd.Timedelta(days=i * step_days) for i in range(folds)]
    folds_out = []
    for fs in fold_starts:
        val_start = fs
        val_end = fs + pd.Timedelta(days=val_len_days)
        if val_end > end_rest:
            val_end = end_rest
            val_start = val_end - pd.Timedelta(days=val_len_days)
        val_mask = (eligible_mask & (dts >= val_start)
                    & (dts <= val_end)).values
        train_cut = val_start - pd.Timedelta(days=purge_days)
        train_mask = (eligible_mask & (dts <= train_cut)).values
        # guard: ensure there is some train data
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds_out.append({
            "val_start": val_start, "val_end": val_end,
            "train_mask": train_mask, "val_mask": val_mask
        })
    if not folds_out:
        # final fallback: single end fold
        val_start = end_rest - pd.Timedelta(days=val_len_days)
        val_end = end_rest
        val_mask = (eligible_mask & (dts >= val_start)
                    & (dts <= val_end)).values
        train_mask = (eligible_mask & (
            dts < (val_start - pd.Timedelta(days=purge_days)))).values
        folds_out = [{"val_start": val_start, "val_end": val_end,
                      "train_mask": train_mask, "val_mask": val_mask}]
    return folds_out


def last_rolling_fold_masks(
    dts: pd.Series,
    eligible_mask: np.ndarray,
    folds: int,
    val_len_days: int,
    purge_days: int
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convenience: return (train_mask, val_mask, info) for the LAST fold.
    """
    flds = build_rolling_folds(
        dts, eligible_mask, folds, val_len_days, purge_days)
    if not flds:
        return eligible_mask, (~eligible_mask), {"val_start": None, "val_end": None, "folds": []}
    last = flds[-1]
    info = {"val_start": last["val_start"],
            "val_end": last["val_end"], "folds": flds}
    return last["train_mask"], last["val_mask"], info
