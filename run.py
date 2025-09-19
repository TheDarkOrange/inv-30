import os
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project imports
from src.config import parse_args
from src.utils import ensure_dir, set_seed, count_trainable
from src.data import top_n_sp500
from src.features import build_panel
from src.model import RNNRegressor
from src.train import (
    panel_to_sequences, MarketDataset,
    compute_scaler_from_train, apply_scaler,
    train_model, evaluate, EMA
)
from src.predict import calibrate_conformal


# -------- helpers --------

def _to_day_ord_numpy(dt_array) -> np.ndarray:
    return np.array(dt_array, dtype="datetime64[D]").astype(np.int64)


def _save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _make_prediction_sequences_for_today(panel: pd.DataFrame, feats: List[str], window: int, t2i: dict) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Build one inference sequence per symbol from the most recent `window` trading days.
    Returns:
        X_pred:   [S, window, F]
        ti_pred:  [S]
        syms:     list of symbols in same order
        last_dt:  [S] numpy datetime64 (last date used per symbol)
    """
    Xs, tis, syms, lastdt = [], [], [], []
    for sym, g in panel.groupby("Symbol", sort=False):
        g = g.sort_values("Date")
        # Keep only rows that have all features (avoid NaNs)
        gi = g[["Date"] + feats].dropna().copy()
        if len(gi) < window:
            continue
        gi = gi.tail(window)
        x = gi[feats].values
        if np.isnan(x).any():
            continue
        Xs.append(x.astype(np.float32))
        tis.append(t2i.get(sym, 0))
        syms.append(sym)
        lastdt.append(pd.to_datetime(gi["Date"].iloc[-1]).to_datetime64())
    if not Xs:
        return (np.zeros((0, window, len(feats)), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                [],
                np.array([], dtype="datetime64[ns]"))
    X_pred = np.stack(Xs).astype(np.float32)
    ti_pred = np.array(tis, dtype=np.int64)
    last_dt = np.array(lastdt, dtype="datetime64[ns]")
    return X_pred, ti_pred, syms, last_dt


def main():
    t0 = time.time()

    # -------- args / defaults for this script --------
    cfg = parse_args()
    # Override a couple of important defaults for this "run" script:
    if not hasattr(cfg, "window") or cfg.window is None:
        cfg.window = 90
    if not hasattr(cfg, "horizon") or cfg.horizon is None:
        cfg.horizon = 30
    if not hasattr(cfg, "m") or cfg.m is None:
        cfg.m = 1
    if not hasattr(cfg, "k") or cfg.k is None:
        cfg.k = 10

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)

    # -------- tickers --------
    print("Selecting top tickers...")
    symbols = top_n_sp500(cfg.n)
    print(f"Top {cfg.n} symbols:", symbols[:10], "...")

    # -------- dates --------
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(cfg.history_years))

    # -------- build panel (features + targets) --------
    print("Downloading & building panel...")
    panel = build_panel(
        symbols, start=start, end=end,
        horizon=int(cfg.horizon),
        use_adj_close=bool(cfg.use_adj_close),
        extra_features=bool(cfg.extra_features),
        allow_download=not bool(cfg.no_download)
    )
    if panel.empty:
        raise RuntimeError("No panel data built. Network/ticker issue?")

    # Train on ABSOLUTE forward log-returns for this script
    target_col = "target_log"
    if target_col not in panel.columns:
        raise RuntimeError("target_log missing from panel.")

    # numeric features only
    drop_cols = {"Date", "Symbol", "target_log", "alpha_log", "bench_log"}
    feats = [c for c in panel.columns if c not in drop_cols and np.issubdtype(
        panel[c].dtype, np.number)]
    if not feats:
        raise RuntimeError("No numeric features discovered for training.")

    # -------- sequences for training/validation --------
    X, y, ti, dates, feats, t2i = panel_to_sequences(
        panel, feats, window=int(cfg.window), target_col=target_col)
    if len(y) == 0:
        raise RuntimeError(
            "No sequences created. Increase history or reduce window.")
    dts = pd.to_datetime(pd.Series(dates))

    # Split: VALIDATION = oldest year; TRAIN = everything else (most recent included)
    dmin = dts.min()
    val_end = dmin + pd.DateOffset(years=1)
    mva = (dts <= val_end).values
    mtr = ~mva

    # Winsorize y on TRAIN only (robustify tails)
    winsor_pct = 0.01
    if mtr.any():
        q = float(np.quantile(np.abs(y[mtr]), 1 - winsor_pct))
    else:
        q = float(np.quantile(np.abs(y), 1 - winsor_pct))
    print(f"Winsorized LOG targets at ±{q:.3f} (train | pct={winsor_pct:.3f})")
    y = np.clip(y, -q, q)

    # Standardize features using TRAIN subset only
    mu, sd = compute_scaler_from_train(X, mtr)
    X = apply_scaler(X, mu, sd)

    # Datasets/loaders
    day_ord_all = _to_day_ord_numpy(dates)
    ds_tr = MarketDataset(X[mtr], y[mtr], ti[mtr],
                          dates[mtr], day_ord_all[mtr])
    ds_va = MarketDataset(X[mva], y[mva], ti[mva],
                          dates[mva], day_ord_all[mva])

    tr_loader = DataLoader(ds_tr, batch_size=int(cfg.batch_size), shuffle=True,
                           num_workers=2, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(ds_va, batch_size=int(cfg.batch_size), shuffle=False,
                           num_workers=2, pin_memory=True, persistent_workers=True)

    # -------- model / ensemble training --------
    ticker_count = len(t2i)
    print(f"Trainable samples: train={len(ds_tr)} val={len(ds_va)}")
    print("Training on raw forward log-returns (ABSOLUTE).")

    ens_val_preds = []
    y_val_ref = None
    models = []

    for k in range(int(cfg.m)):
        print(f"\n=== Ensemble member {k+1}/{cfg.m} ===")
        set_seed(cfg.seed + k)
        model = RNNRegressor(
            rnn_type=cfg.rnn_type,
            n_features=X.shape[2],
            hidden_size=int(cfg.hidden_size),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
            n_tickers=ticker_count,
            ticker_embed_dim=16
        ).to(device)

        # optional torch.compile
        if int(cfg.compile) and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, dynamic=False, mode="default")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(
                    f"torch.compile failed ({e}); continuing without compile.")

        print(f"Trainable parameters: {count_trainable(model):,}")

        ema = EMA(model, decay=0.995)
        model = train_model(
            model, tr_loader, va_loader, device,
            epochs=int(cfg.epochs), lr=float(cfg.lr), amp=bool(cfg.amp),
            weight_decay=1e-3, clip_grad=1.0,
            lr_plateau_patience=1, lr_plateau_factor=0.5, lr_min=1e-5,
            early_stop_patience=3, noise_std=0.02, rank_weight=float(cfg.rank_weight),
            time_decay_half_life=365, max_train_ordinal=int(day_ord_all[mtr].max()),
            ema=ema, loss_type=cfg.loss, huber_delta=float(cfg.huber_delta),
            rank_loss=cfg.rank_loss, listwise_temp=float(cfg.listwise_temp)
        )

        pv, yv, val_metrics = evaluate(
            model, va_loader, device, amp=bool(cfg.amp))
        if y_val_ref is None:
            y_val_ref = yv
        ens_val_preds.append(pv.reshape(-1, 1))
        models.append(model)

    # Ensemble mean/std on validation for conformal calibration
    P_val = np.concatenate(
        ens_val_preds, axis=1) if ens_val_preds else np.zeros((len(ds_va), 1))
    pred_va_mean = P_val.mean(axis=1) if P_val.size else np.zeros(
        (0,), dtype=np.float32)

    # Conformal interval (90%)
    conf = calibrate_conformal(pred_va_mean, y_val_ref, alpha=0.1)
    qhat = float(conf.get("qhat", 0.0))
    print("Calibrating conformal intervals on validation...")

    # -------- build "today" sequences (last `window` trading days) --------
    # IMPORTANT: Use the same features list `feats`, and scale with (mu, sd)
    Xp, tip, syms, last_dt = _make_prediction_sequences_for_today(
        panel, feats, int(cfg.window), t2i)
    if Xp.shape[0] == 0:
        raise RuntimeError(
            "No symbols have at least `window` recent rows for inference.")
    Xp = apply_scaler(Xp, mu, sd)

    # Batch predict for each model
    with torch.no_grad():
        preds_all = []
        for mdl in models:
            mdl.eval()
            # chunk to avoid OOM on big N
            chunk = 1024
            out = []
            for i in range(0, Xp.shape[0], chunk):
                xb = torch.from_numpy(Xp[i:i+chunk]).to(device)
                tb = torch.from_numpy(tip[i:i+chunk]).to(device)
                mu_hat = mdl(xb, tb).detach().cpu().numpy().reshape(-1)
                out.append(mu_hat)
            preds_all.append(np.concatenate(out, axis=0).reshape(-1, 1))
        P_pred = np.concatenate(
            preds_all, axis=1) if preds_all else np.zeros((Xp.shape[0], 1))

    pred_mean_log = P_pred.mean(axis=1)
    pred_std_log = P_pred.std(axis=1, ddof=0)
    # Conformal 90% interval in log space
    lo_log = pred_mean_log - qhat
    hi_log = pred_mean_log + qhat

    # Convert to ROI (%)
    pred_mean_roi = np.exp(pred_mean_log) - 1.0
    lo_roi = np.exp(lo_log) - 1.0
    hi_roi = np.exp(hi_log) - 1.0

    # Collect results
    res = pd.DataFrame({
        "Symbol": syms,
        "LastDateUsed": pd.to_datetime(last_dt).normalize(),
        "Pred_Log": pred_mean_log,
        "Pred_ROI": pred_mean_roi,
        "Ensemble_STD_Log": pred_std_log,
        "CI90_Low_Log": lo_log,
        "CI90_High_Log": hi_log,
        "CI90_Low_ROI": lo_roi,
        "CI90_High_ROI": hi_roi
    }).sort_values("Pred_ROI", ascending=False).reset_index(drop=True)

    # Print top K
    K = int(cfg.k)
    print(
        f"\nTop {K} predicted 30-day returns (window={cfg.window} trading days up to today):")
    for i, row in res.head(K).iterrows():
        sym = row["Symbol"]
        roi = float(row["Pred_ROI"])
        lo = float(row["CI90_Low_ROI"])
        hi = float(row["CI90_High_ROI"])
        std = float(row["Ensemble_STD_Log"])
        dt = pd.to_datetime(row["LastDateUsed"]).date()
        print(
            f"{i+1:02d}. {sym:<6s} | pred: {roi:+.2%} | 90% CI: [{lo:+.2%}, {hi:+.2%}] | std(log): {std:.4f} | last_date: {dt}")

    # Save CSV
    ensure_dir("artifacts")
    out_csv = "artifacts/today_predictions.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Done in {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()
