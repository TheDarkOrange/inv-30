from __future__ import annotations
import os
import time
from typing import Dict, List
import pandas as pd
import numpy as np
import yfinance as yf
from .utils import ensure_dir, today_str_local

SYMS_CSV = "config/sp500_by_mcap.csv"


def top_n_sp500(n: int) -> List[str]:
    df = pd.read_csv(SYMS_CSV)
    syms = df["Symbol"].astype(str).str.strip().tolist()
    return syms[:n]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # strip + keep original case; leave rest to features’ tolerant accessors
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _dl_one(sym: str, start, end) -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(sym, start=start, end=end,
                             progress=False, auto_adjust=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                # flatten last level: ('Adj Close', 'SPY') -> 'Adj Close'
                df.columns = [
                    c[-1] if isinstance(c, tuple) else c for c in df.columns]
            df = _normalize_cols(df)
            df = df.rename_axis("Date").reset_index().set_index("Date")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    # fallthrough
    return pd.DataFrame()


def _has_price_cols(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip().lower().replace(" ", "") for c in df.columns}
    return any(k in cols for k in ("adjclose", "close"))


def load_or_download_price_history(symbols: List[str], start, end, cache_dir: str = "data", allow_download: bool = True) -> Dict[str, pd.DataFrame]:
    ensure_dir(cache_dir)
    today = today_str_local()
    out: Dict[str, pd.DataFrame] = {}
    # warmup pad for indicators
    pad_start = pd.to_datetime(start) - pd.Timedelta(days=400)
    pad_end = pd.to_datetime(end) + pd.Timedelta(days=10)

    def _read_cached(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        df = _normalize_cols(df)
        df.index = pd.to_datetime(df.index)
        return df

    for s in symbols:
        path = os.path.join(cache_dir, f"{s}.csv")
        df_cached = _read_cached(path)
        has_price_cols = _has_price_cols(df_cached)

        need_dl = False
        if allow_download:
            if not has_price_cols:
                need_dl = True
            else:
                if os.path.exists(path):
                    file_day = pd.to_datetime(os.path.getmtime(
                        path), unit='s').date().isoformat()
                    if file_day != today:
                        need_dl = True
                else:
                    need_dl = True

        if need_dl:
            df_new = _dl_one(s, pad_start, pad_end)
            if not df_new.empty:
                df_new.to_csv(path)
            df_cached = _read_cached(path)

        if allow_download and (df_cached.empty or not _has_price_cols(df_cached)):
            df_new = _dl_one(s, pad_start, pad_end)
            if not df_new.empty:
                df_new.to_csv(path)
            df_cached = _read_cached(path)

        if df_cached is None or df_cached.empty:
            df_final = pd.DataFrame()
        else:
            df_final = df_cached.sort_index()
            df_final = df_final[(df_final.index >= pad_start)
                                & (df_final.index <= pad_end)]

        out[s] = df_final
    return out
