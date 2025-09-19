import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import parse_args
from src.utils import ensure_dir, set_seed, count_trainable
from src.data import top_n_sp500
from src.features import build_panel
from src.model import RNNRegressor
from src.train import panel_to_sequences, MarketDataset, compute_scaler_from_train, apply_scaler, train_model, evaluate, EMA
from src.predict import calibrate_conformal
from src.backtest import step_pick_best_log, save_backtest_plot


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _to_day_ord_numpy(dt_array) -> np.ndarray:
    return np.array(dt_array, dtype="datetime64[D]").astype(np.int64)


def main():
    t0 = time.time()
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)

    # 1) Symbols
    print("Selecting top tickers...")
    symbols = top_n_sp500(cfg.n)
    print(f"Top {cfg.n} symbols:", symbols[:10], "...")

    # 2) Date span
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(cfg.history_years))

    # 3) Build panel
    print("Downloading & building panel...")
    panel = build_panel(
        symbols, start=start, end=end, horizon=cfg.horizon,
        use_adj_close=bool(cfg.use_adj_close), extra_features=bool(cfg.extra_features),
        allow_download=not bool(cfg.no_download)
    )
    if panel.empty:
        raise RuntimeError("No panel data built. Check tickers/network.")
    target_col = "alpha_log" if "alpha_log" in panel.columns else "target_log"

    # Features
    drop_cols = {"Date", "Symbol", "target_log", "alpha_log", "bench_log"}
    feats = [c for c in panel.columns if c not in drop_cols and np.issubdtype(
        panel[c].dtype, np.number)]

    # 4) Sequences
    X, y, ti, dates, feats, t2i = panel_to_sequences(
        panel, feats, window=int(cfg.window), target_col=target_col)
    if len(y) == 0:
        raise RuntimeError(
            "No sequences created. Try increasing history or reducing window.")

    # 5) Split: Test = last t months; Train/Val = random split of the rest
    dts = pd.to_datetime(pd.Series(dates))

    t_months = int(getattr(cfg, "t_recent_months", 6))
    cutoff = dts.max() - pd.DateOffset(months=t_months)
    mte = (dts >= cutoff).values

    # Fallback if (somehow) no test sequences exist
    if mte.sum() == 0:
        fallback_cutoff = dts.max() - pd.DateOffset(years=1)
        mte = (dts >= fallback_cutoff).values
        print(f"[warn] No samples in last {t_months} months; falling back to last 12 months "
              f"({fallback_cutoff.date()} → {dts.max().date()}).")
    else:
        print(f"Test period: last {t_months} months "
              f"({cutoff.date()} → {dts.max().date()}).")

    idx_rest = np.where(~mte)[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(idx_rest)
    n_val = max(1, int(len(idx_rest) * 0.15))
    val_set = set(perm[:n_val].tolist())
    train_set = set(perm[n_val:].tolist())
    mtr = np.array([i in train_set for i in range(len(dates))], dtype=bool)
    mva = np.array([i in val_set for i in range(len(dates))], dtype=bool)

    # Winsorize y on TRAIN only
    winsor_pct = 0.01
    q = float(np.quantile(np.abs(y[mtr]), 1 - winsor_pct)
              ) if mtr.any() else float(np.quantile(np.abs(y), 1 - winsor_pct))
    print(f"Winsorized LOG targets at ±{q:.3f} (train | pct={winsor_pct:.3f})")
    y = np.clip(y, -q, q)

    # scale X using train subset
    mu, sd = compute_scaler_from_train(X, mtr)
    X = apply_scaler(X, mu, sd)

    # day ordinals without .view()
    day_ord_all = _to_day_ord_numpy(dates)

    # 6) Datasets / loaders
    ds_tr = MarketDataset(X[mtr], y[mtr], ti[mtr],
                          dates[mtr], day_ord_all[mtr])
    ds_va = MarketDataset(X[mva], y[mva], ti[mva],
                          dates[mva], day_ord_all[mva])
    ds_te = MarketDataset(X[mte], y[mte], ti[mte],
                          dates[mte], day_ord_all[mte])

    tr_loader = DataLoader(ds_tr, batch_size=int(
        cfg.batch_size), shuffle=True,  num_workers=2, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(ds_va, batch_size=int(
        cfg.batch_size), shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    te_loader = DataLoader(ds_te, batch_size=int(
        cfg.batch_size), shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # 7) Model + Ensemble
    ticker_count = len(t2i)
    i2t = {i: t for t, i in t2i.items()}
    print(
        f"Trainable samples: train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
    print("Training on ALPHA (stock logret - benchmark logret)..." if target_col ==
          "alpha_log" else "Training on raw forward log-returns...")

    ens_val_preds, ens_test_preds = [], []
    y_val_ref, y_test_ref = None, None

    for k in range(int(cfg.m)):
        print(f"\n=== Ensemble member {k+1}/{cfg.m} ===")
        set_seed(cfg.seed + k)
        model = RNNRegressor(
            rnn_type=cfg.rnn_type, n_features=X.shape[2], hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, dropout=cfg.dropout, n_tickers=ticker_count, ticker_embed_dim=16
        ).to(device)

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
            weight_decay=1e-3, clip_grad=1.0, lr_plateau_patience=1, lr_plateau_factor=0.5, lr_min=1e-5,
            early_stop_patience=5, noise_std=0.02, rank_weight=float(cfg.rank_weight),
            time_decay_half_life=365, max_train_ordinal=int(day_ord_all[mtr].max()),
            ema=ema, loss_type=cfg.loss, huber_delta=float(cfg.huber_delta),
            rank_loss=cfg.rank_loss, listwise_temp=float(cfg.listwise_temp)
        )

        pv, yv, val_metrics = evaluate(
            model, va_loader, device, amp=bool(cfg.amp))
        pt, yt, test_metrics = evaluate(
            model, te_loader, device, amp=bool(cfg.amp))

        if y_val_ref is None:
            y_val_ref = yv
        if y_test_ref is None:
            y_test_ref = yt
        ens_val_preds.append(pv.reshape(-1, 1))
        ens_test_preds.append(pt.reshape(-1, 1))

    # Aggregate ensemble
    P_val = np.concatenate(
        ens_val_preds, axis=1) if ens_val_preds else np.zeros((len(ds_va), 1))
    P_te = np.concatenate(
        ens_test_preds, axis=1) if ens_test_preds else np.zeros((len(ds_te), 1))
    pred_va_mean = P_val.mean(axis=1)
    pred_va_std = P_val.std(axis=1, ddof=0)
    pred_te_mean = P_te.mean(axis=1)
    pred_te_std = P_te.std(axis=1, ddof=0)

    # Blender (optional)
    if int(cfg.blend_ridge):
        from sklearn.linear_model import Ridge
        X_last = X[:, -1, :]
        val_idx = np.where(~mte & ~mtr)[0]  # indices used for val; but easier:
        Xv = X[mva][:, -1, :]
        Xt = X[mte][:, -1, :]
        R = Ridge(alpha=float(cfg.blend_alpha))
        R.fit(np.hstack([pred_va_mean.reshape(-1, 1), Xv]), y_val_ref)
        pred_va_mean = R.predict(np.hstack([pred_va_mean.reshape(-1, 1), Xv]))
        pred_te_mean = R.predict(np.hstack([pred_te_mean.reshape(-1, 1), Xt]))

    # Conformal calibration
    conf = calibrate_conformal(pred_va_mean, y_val_ref, alpha=0.1)
    print("Calibrating conformal intervals on validation...")

    # 8) Build prediction DataFrame for test set
    test_idx = np.where(mte)[0]
    rows = []
    for j, idx in enumerate(test_idx):
        mu = float(pred_te_mean[j])
        sd = float(pred_te_std[j])
        lo, hi = mu - conf.get("qhat", 0.0), mu + conf.get("qhat", 0.0)
        rows.append({
            "Date": pd.to_datetime(dates[idx]).normalize(),
            "Symbol": i2t[int(ti[idx])] if int(ti[idx]) in i2t else str(ti[idx]),
            "pred_log_mean": mu, "pred_log_std": sd,
            "target_log": float(y[idx]),
            "lo_log": float(lo), "hi_log": float(hi),
        })
    pred_df = pd.DataFrame(rows).sort_values(
        ["Date", "Symbol"]).reset_index(drop=True)

    # 9) Backtest
    bt = step_pick_best_log(
        pred_df,
        horizon=int(cfg.horizon),
        benchmark_symbol=cfg.benchmark_symbol,
        use_adj_close=bool(cfg.use_adj_close),
        k=int(cfg.k),
        lambda_conf=float(cfg.lambda_conf),
        weighting=cfg.weighting,
        softmax_temp=float(cfg.softmax_temp),
        hold_if_uncertain=bool(cfg.hold_if_uncertain),
    )

    # 10) Save artifacts
    ensure_dir("artifacts")
    pred_df.to_csv(
        "artifacts/test_predictions_with_confidence.csv", index=False)
    bt.to_csv("artifacts/backtest.csv", index=False)
    from src.backtest import save_backtest_plot
    save_backtest_plot(bt, "artifacts/backtest_equity.png",
                       title="Equity Curve (Strategy vs S&P)")
    print("Saved backtest plot to artifacts/backtest_equity.png")

    # Per-period ROI + Sharpe
    print("\nBacktest summary (horizon = {} days)".format(cfg.horizon))
    if not bt.empty:
        model_eq = 1.0
        bench_eq = 1.0
        for _, r in bt.iterrows():
            m_roi = float(r["Realized"]) if pd.notna(r["Realized"]) else 0.0
            b_roi = float(r["BenchROI"]) if pd.notna(r["BenchROI"]) else 0.0
            model_eq *= (1.0 + m_roi)
            bench_eq *= (1.0 + b_roi)
            print(
                f"{pd.to_datetime(r['Date']).date()} | Picked: {r['Picked']:<20s} | ModelROI: {m_roi:+.4f} | S&P: {b_roi:+.4f} | Cum(Model): {model_eq:.6f} | Cum(S&P): {bench_eq:.6f}")

        periods_per_year = 365.0 / float(cfg.horizon)

        def sharpe(x):
            x = x[~np.isnan(x)]
            return float((np.sqrt(periods_per_year) * x.mean() / (x.std(ddof=1) + 1e-9)) if x.size > 1 else np.nan)
        rets = bt["Realized"].astype(float).to_numpy()
        brets = bt["BenchROI"].astype(float).fillna(0.0).to_numpy()
        print(
            f"\nProducts → Model: {model_eq:.6f} (Total: {(model_eq-1):.2%}) | S&P: {bench_eq:.6f} (Total: {(bench_eq-1):.2%})")
        print(
            f"Sharpe → Strategy: {sharpe(rets):.3f} | S&P: {sharpe(brets):.3f}")
    else:
        print("No trades produced.")

    print(f"Done in {time.time()-t0:.2f}s. Artifacts in ./artifacts")


if __name__ == "__main__":
    main()
