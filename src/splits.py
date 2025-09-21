from typing import Tuple
import numpy as np
import pandas as pd


def build_train_val_masks(
    dts: pd.Series,
    scheme: str = "oldest_year",
    cutoff: pd.Timestamp = None,   # exclusive upper bound for train/val
    wf_val_months: int = 1,
    wf_stride_months: int = 1,
    wf_max_splits: int = 6,
    seed: int = 42,
    purge_days: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    dts = pd.to_datetime(pd.Series(dts))
    N = len(dts)
    mtr = np.zeros(N, dtype=bool)
    mva = np.zeros(N, dtype=bool)

    if cutoff is None:
        cutoff = dts.max()
    pre_mask = (dts <= cutoff)

    if scheme == "oldest_year":
        if pre_mask.sum() == 0:
            return mtr, mva
        dmin = dts[pre_mask].min()
        val_end = dmin + pd.DateOffset(years=1)
        mva = (dts <= val_end) & pre_mask
        mtr = pre_mask & (~mva)

    elif scheme == "walk_forward":
        val_mask = np.zeros(N, dtype=bool)
        purge_mask = np.zeros(N, dtype=bool)
        end = pd.to_datetime(cutoff)
        taken = 0
        purge_td = pd.Timedelta(days=int(max(0, purge_days)))

        while taken < int(wf_max_splits):
            start = end - pd.DateOffset(months=int(wf_val_months))
            block = (dts > start) & (dts <= end) & pre_mask
            if block.sum() == 0:
                if (dts[pre_mask].min() >= start):
                    break
            val_mask |= block

            if purge_td.value > 0:
                pre_purge = (dts > (start - purge_td)
                             ) & (dts <= start) & pre_mask
                post_purge = (dts > end) & (dts <= (end + purge_td)) & pre_mask
                purge_mask |= (pre_purge | post_purge)

            taken += 1
            end = start - pd.DateOffset(months=int(wf_stride_months))
            if end <= dts[pre_mask].min():
                break

        mva = val_mask
        mtr = pre_mask & (~val_mask) & (~purge_mask)

    else:
        raise ValueError(f"Unknown val scheme: {scheme}")

    return mtr, mva


def random_split_pretest(dts: pd.Series, cutoff: pd.Timestamp, val_frac: float = 0.15, seed: int = 42):
    dts = pd.to_datetime(pd.Series(dts))
    pre_idx = np.where(dts <= cutoff)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pre_idx)
    n_val = max(1, int(len(pre_idx) * val_frac)) if len(pre_idx) > 0 else 0
    val_idx = set(perm[:n_val].tolist())
    mva = np.array([i in val_idx for i in range(len(dts))], dtype=bool)
    mtr = (dts <= cutoff).values & (~mva)
    return mtr, mva
