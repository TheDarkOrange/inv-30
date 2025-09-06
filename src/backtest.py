import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from .data import load_or_download_price_history

# ---------- tolerant price access ----------


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)

    # exact match first
    for c in candidates:
        if c in cols:
            return c

    # case/space-insensitive
    norm = {str(c): str(c).strip().lower().replace(" ", "") for c in cols}
    wanted = [c.strip().lower().replace(" ", "") for c in candidates]
    for k, v in norm.items():
        if v in wanted:
            return k

    # substring fallback
    for want in wanted:
        for k, v in norm.items():
            if want in v:
                return k
    return None


def _get(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    col = _find_col(df, names)
    return df[col].astype(float) if col is not None else None


def _px(df: pd.DataFrame, use_adj: bool = True) -> pd.Series:
    if df is None or df.empty:
        raise KeyError("Empty DataFrame for price extraction.")
    if use_adj:
        s = _get(df, ["Adj Close", "AdjClose", "Adj_Close"])
        if s is not None:
            return s
    s = _get(df, ["Close"])
    if s is not None:
        return s
    # final fallback: first numeric column
    for c in df.columns:
        try:
            return df[c].astype(float)
        except Exception:
            continue
    raise KeyError("No suitable price column found (Adj Close/Close).")

# ---------- helpers ----------


def _prep_benchmark_series(symbol: str, start, end, use_adj: bool = True):
    data = load_or_download_price_history([symbol], start=start, end=end)
    df = data.get(symbol, pd.DataFrame())
    if df is None or df.empty:
        return None, None
    idx = pd.to_datetime(df.index)
    px = _px(df, use_adj=use_adj).astype(float).to_numpy()
    return idx.to_numpy(), px


def _prep_symbol_maps(symbols, start, end, use_adj: bool = True):
    data = load_or_download_price_history(list(symbols), start=start, end=end)
    maps = {}
    for s, df in data.items():
        if df is None or df.empty:
            continue
        idx = pd.to_datetime(df.index).to_numpy()
        px = _px(df, use_adj=use_adj).astype(float).to_numpy()
        maps[s] = (idx, px)
    return maps


def _price_and_date_on_or_after(dates_np, prices_np, ts: np.datetime64):
    i = dates_np.searchsorted(ts, side="left")
    if i >= len(dates_np):
        return np.nan, None
    p = np.asarray(prices_np).reshape(-1)[i]
    d = dates_np[i]
    p = float(p.item() if hasattr(p, "item") else p)
    return p, pd.Timestamp(d).normalize()

# ---------- main backtest ----------


def step_pick_best_log(pred_df: pd.DataFrame, horizon: int = 30, benchmark_symbol: str = "SPY",
                       use_adj_close: bool = True, k: int = 1, lambda_conf: float = 1.0,
                       weighting: str = 'equal', softmax_temp: float = 1.0, hold_if_uncertain: bool = True) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.tz_localize(None)
    pred_df = pred_df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    uniq_dates = sorted(pred_df["Date"].unique().tolist())
    if not uniq_dates:
        return pd.DataFrame(columns=["Date", "Picked", "PredLog", "Realized", "BuyDate", "BuyPrice", "SellDate", "SellPrice", "Equity", "BenchROI", "BenchEquity"])

    # decision dates separated by at least `horizon` days
    decisions = []
    i = 0
    d = uniq_dates[0]
    last = uniq_dates[-1]
    while d <= last:
        while i < len(uniq_dates) and uniq_dates[i] < d:
            i += 1
        if i >= len(uniq_dates):
            break
        decision_day = uniq_dates[i]
        if len(decisions) == 0 or decision_day > decisions[-1]:
            decisions.append(decision_day)
        d = decision_day + pd.Timedelta(days=int(horizon))

    # prepare benchmark series
    bench_dates_np, bench_prices_np = (None, None)
    if benchmark_symbol and len(decisions) > 0:
        start = pd.Timestamp(min(decisions)) - pd.Timedelta(days=horizon + 10)
        end = pd.Timestamp(max(decisions)) + pd.Timedelta(days=horizon + 10)
        bench_dates_np, bench_prices_np = _prep_benchmark_series(
            benchmark_symbol, start, end, use_adj=use_adj_close)

    # prepare symbol maps
    chosen_syms = set(pred_df["Symbol"].unique().tolist())
    sym_maps = {}
    if len(decisions) > 0 and len(chosen_syms) > 0:
        start = pd.Timestamp(min(decisions)) - pd.Timedelta(days=horizon + 10)
        end = pd.Timestamp(max(decisions)) + pd.Timedelta(days=horizon + 10)
        sym_maps = _prep_symbol_maps(
            chosen_syms, start, end, use_adj=use_adj_close)

    equity = 1.0
    bench_equity = 1.0
    rows = []
    for d in decisions:
        sl = pred_df[pred_df["Date"] == d]
        if sl.empty:
            continue
        use_col = "pred_log_mean" if "pred_log_mean" in sl.columns else "pred_log"
        std_col = "pred_log_std" if "pred_log_std" in sl.columns else None
        scores = sl[use_col] - (lambda_conf * sl[std_col]
                                ) if std_col else sl[use_col]
        sl = sl.assign(score=scores).sort_values("score", ascending=False)

        confident = (len(sl) > 0 and float(sl.iloc[0]["score"]) > 0)
        pick_list = []
        if confident:
            sel = sl.head(max(1, int(k)))
            if weighting == "softmax":
                w_raw = np.exp(sel["score"].values /
                               max(1e-6, float(softmax_temp)))
                w = w_raw / w_raw.sum()
            else:
                w = [1.0 / len(sel)] * len(sel)
            for (idx_row, row), wi in zip(sel.iterrows(), w):
                pick_list.append((row["Symbol"], wi, row[use_col]))

        # Benchmark leg
        bench_roi = None
        if bench_dates_np is not None:
            p0b, _ = _price_and_date_on_or_after(
                bench_dates_np, bench_prices_np, pd.Timestamp(d).to_datetime64())
            p1b, _ = _price_and_date_on_or_after(bench_dates_np, bench_prices_np, (pd.Timestamp(
                d) + pd.Timedelta(days=horizon)).to_datetime64())
            if np.isfinite(p0b) and np.isfinite(p1b) and p0b > 0:
                bench_roi = p1b / p0b - 1.0
                bench_equity *= (1.0 + bench_roi)

        # Strategy leg (portfolio aggregation)
        real_simple = 0.0
        buys = []
        sells = []
        if confident and len(pick_list) > 0:
            for sym_i, wi, pred_i in pick_list:
                buy_p = sell_p = np.nan
                buy_dt = sell_dt = None
                realized_log = np.nan
                if sym_i in sym_maps:
                    dates_np, prices_np = sym_maps[sym_i]
                    buy_p, buy_dt = _price_and_date_on_or_after(
                        dates_np, prices_np, pd.Timestamp(d).to_datetime64())
                    sell_p, sell_dt = _price_and_date_on_or_after(
                        dates_np, prices_np, (pd.Timestamp(d) + pd.Timedelta(days=horizon)).to_datetime64())
                    if np.isfinite(buy_p) and np.isfinite(sell_p) and buy_p > 0:
                        realized_log = float(np.log(sell_p / buy_p))
                if not np.isfinite(realized_log):
                    # fallback from panel targets
                    realized_log = float(
                        sl[sl["Symbol"] == sym_i]["target_log"].iloc[0])
                r_simple = float(np.exp(realized_log) - 1.0)
                real_simple += wi * r_simple
                buys.append((sym_i, buy_dt, float(buy_p)
                            if np.isfinite(buy_p) else None, wi))
                sells.append((sym_i, sell_dt, float(sell_p)
                             if np.isfinite(sell_p) else None, wi))
        else:
            real_simple = 0.0  # hold/skip

        equity *= (1.0 + real_simple)
        picks_str = ",".join([f"{s}:{w:.3f}" for s, _, __, w in buys]) if len(
            buys) > 0 else ("HOLD_SPY" if not confident else "")
        rows.append({
            "Date": d.normalize(),
            "Picked": picks_str,
            "PredLog": float(sl.iloc[0][use_col]) if len(sl) > 0 else None,
            "Realized": real_simple,
            "BuyDate": buys[0][1] if len(buys) > 0 else None,
            "BuyPrice": buys[0][2] if len(buys) > 0 else None,
            "SellDate": sells[0][1] if len(sells) > 0 else None,
            "SellPrice": sells[0][2] if len(sells) > 0 else None,
            "Equity": equity,
            "BenchROI": float(bench_roi) if bench_roi is not None else None,
            "BenchEquity": float(bench_equity) if bench_roi is not None else None
        })
    return pd.DataFrame(rows)


def save_backtest_plot(bt: pd.DataFrame, outpath: str, title: str = "Equity Curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if bt is None or bt.empty:
        return
    df = bt.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=144)
    ax.plot(df["Date"], df["Equity"], label="Strategy")
    if "BenchEquity" in df.columns and df["BenchEquity"].notna().any():
        ax.plot(df["Date"], df["BenchEquity"], label="S&P (SPY)")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (growth of $1)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
