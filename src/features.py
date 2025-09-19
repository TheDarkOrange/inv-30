from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from .data import load_or_download_price_history


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    norm = {str(c): str(c).strip().lower().replace(" ", "") for c in cols}
    wanted = [c.strip().lower().replace(" ", "") for c in candidates]
    for k, v in norm.items():
        if v in wanted:
            return k
    for want in wanted:
        for k, v in norm.items():
            if want in v:
                return k
    return None


def _get(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    col = _find_col(df, names)
    return df[col].astype(float) if col is not None else None


def _px(df: pd.DataFrame, use_adj=True) -> pd.Series:
    if df is None or df.empty:
        raise KeyError("Empty DataFrame for price extraction")
    if use_adj:
        s = _get(df, ["Adj Close", "AdjClose", "Adj_Close"])
        if s is not None:
            return s
    s = _get(df, ["Close"])
    if s is not None:
        return s
    for c in df.columns:
        try:
            return df[c].astype(float)
        except Exception:
            continue
    raise KeyError("No suitable price column found (Adj Close/Close).")


def technical_features(df, use_adj=True, extra=True):
    idx = pd.to_datetime(df.index)
    close = _px(df, use_adj)
    out = pd.DataFrame(index=idx)
    out["Close"] = close
    out["ret_1"] = close.pct_change()
    out["logret_1"] = np.log(close/close.shift())
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["sma_10"] = close.rolling(10).mean() / (close + 1e-9) - 1.0
    out["sma_20"] = close.rolling(20).mean() / (close + 1e-9) - 1.0
    out["sma_50"] = close.rolling(50).mean() / (close + 1e-9) - 1.0
    out["moy_sin"] = np.sin(2*np.pi*(idx.month.values)/12.0).astype(float)
    out["moy_cos"] = np.cos(2*np.pi*(idx.month.values)/12.0).astype(float)

    if extra:
        out["mom_20"] = out["Close"].pct_change(20)
        out["mom_60"] = out["Close"].pct_change(60)
        out["mom_120"] = out["Close"].pct_change(120)
        high = _get(df, ["High"])
        low = _get(df, ["Low"])
        prev_close = close.shift()
        if high is not None and low is not None:
            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            out["atr_14"] = tr.rolling(14).mean() / (out["Close"] + 1e-9)
        else:
            out["atr_14"] = out["vol_20"]
        roll_max = out["Close"].rolling(120, min_periods=1).max()
        out["drawdown_120"] = (out["Close"] / (roll_max + 1e-9)) - 1.0
        vol = _get(df, ["Volume"])
        if vol is not None:
            out["vol_z"] = (vol - vol.rolling(20).mean()) / \
                (vol.rolling(20).std() + 1e-9)
        else:
            out["vol_z"] = 0.0

    return out.dropna()


def build_panel(symbols: List[str], start, end, horizon, use_adj_close=True, extra_features=True, allow_download: bool = True) -> pd.DataFrame:
    price = load_or_download_price_history(
        symbols, start, end, allow_download=allow_download)
    bench = load_or_download_price_history(
        ["SPY"], start, end, allow_download=allow_download).get("SPY", pd.DataFrame())

    bench_log_fwd = None
    if bench is not None and not bench.empty:
        try:
            bpx = _px(bench, use_adj_close)
            bench_log_fwd = np.log(bpx.shift(-int(horizon)) / bpx)
        except Exception:
            bench_log_fwd = None

    rows = []
    for sym in symbols:
        df = price.get(sym, pd.DataFrame())
        if df is None or df.empty:
            continue
        tf = technical_features(
            df, use_adj=use_adj_close, extra=extra_features)
        px = tf["Close"]
        target_log = np.log(px.shift(-int(horizon)) / px)
        out = tf.copy()
        out["Symbol"] = sym
        out["target_log"] = target_log
        if bench_log_fwd is not None:
            out["bench_log"] = bench_log_fwd.reindex(out.index)
            out["alpha_log"] = out["target_log"] - out["bench_log"]
        rows.append(out.reset_index().rename(columns={"index": "Date"}))

    if not rows:
        return pd.DataFrame(columns=["Date", "Symbol", "target_log"])

    panel = pd.concat(rows, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)
    panel = panel[(panel["Date"] >= pd.to_datetime(start))
                  & (panel["Date"] <= pd.to_datetime(end))]
    panel = panel.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return panel
