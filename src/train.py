from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler


def _to_day_ord_numpy(dt_array) -> np.ndarray:
    return np.array(dt_array, dtype="datetime64[D]").astype(np.int64)


def _weights_time_decay(date_ord: np.ndarray, max_train_ordinal: int, half_life_days: int) -> torch.Tensor:
    delta = max_train_ordinal - date_ord
    w = 0.5 ** (delta / max(1, half_life_days))
    return torch.tensor(w, dtype=torch.float32)


def _huber_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0, reduction: str = "none") -> torch.Tensor:
    diff = input - target
    absdiff = torch.abs(diff)
    d = torch.tensor(delta, device=input.device, dtype=input.dtype)
    quad = torch.minimum(absdiff, d) ** 2
    lin = absdiff - torch.minimum(absdiff, d)
    loss = 0.5 * quad + d * lin
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _rank_top1_loss(logits: torch.Tensor, targets: torch.Tensor, dates_int) -> Optional[torch.Tensor]:
    if dates_int is None:
        return None
    from collections import defaultdict
    group = defaultdict(list)
    for i, d in enumerate(dates_int):
        group[int(d)].append(i)
    losses = []
    for idxs in group.values():
        if len(idxs) < 2:
            continue
        li = logits[idxs].unsqueeze(0)
        ti = targets[idxs].unsqueeze(0)
        diff = (li.transpose(1, 2) - li)
        sign = torch.sign((ti.transpose(1, 2) - ti))
        losses.append(torch.relu(1.0 - diff*sign).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _rank_listwise_loss(logits: torch.Tensor, targets: torch.Tensor, dates_int, temp: float = 1.0) -> Optional[torch.Tensor]:
    if dates_int is None:
        return None
    from collections import defaultdict
    group = defaultdict(list)
    for i, d in enumerate(dates_int):
        group[int(d)].append(i)
    losses = []
    for idxs in group.values():
        if len(idxs) < 2:
            continue
        g_logits = logits[idxs]
        g_targets = targets[idxs]
        p = torch.softmax(g_logits / max(1e-6, temp), dim=0)
        t = torch.softmax(g_targets / max(1e-6, temp), dim=0)
        losses.append(F.kl_div(p.log(), t, reduction='batchmean'))
    if not losses:
        return None
    return torch.stack(losses).mean()


class MarketDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ticker_ids: np.ndarray, dates: np.ndarray, date_ord: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.ti = ticker_ids.astype(np.int64)
        self.dates = dates
        self.date_ord = date_ord.astype(np.int64)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.ti[i], self.date_ord[i]


def panel_to_sequences(panel: pd.DataFrame, features: List[str], window: int, target_col: str) -> Tuple[np.ndarray, ...]:
    panel = panel.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    tickers = sorted(panel["Symbol"].unique().tolist())
    t2i = {t: i for i, t in enumerate(tickers)}

    Xs, ys, tis, dts = [], [], [], []
    for sym, g in panel.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)

        # Ensure all requested features exist for this symbol
        if any(f not in g.columns for f in features):
            continue

        # Positional, no label-based alignment (pre-extract NumPy views)
        A = g[features].to_numpy(
            dtype=np.float32, copy=False)            # [T, F]
        tgt = g[target_col].to_numpy(
            dtype=np.float32, copy=False)          # [T]
        dta = pd.to_datetime(g["Date"]).to_numpy(
            dtype="datetime64[ns]")    # [T]

        T = len(g)
        if T < window:
            continue

        for i in range(window - 1, T):
            y_i = tgt[i]
            if not np.isfinite(y_i):
                continue
            x_i = A[i - (window - 1): i + 1, :]  # strictly positional window
            if not np.isfinite(x_i).all():
                continue

            Xs.append(x_i)
            ys.append(float(y_i))
            tis.append(t2i[sym])
            dts.append(dta[i])

    if not Xs:
        X = np.zeros((0, window, len(features)), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32)
        ti = np.zeros((0,), dtype=np.int64)
        dates = np.array([], dtype="datetime64[ns]")
        date_ord = np.array([], dtype=np.int64)
    else:
        X = np.stack(Xs).astype(np.float32)
        y = np.array(ys, dtype=np.float32)
        ti = np.array(tis, dtype=np.int64)
        dates = np.array(dts, dtype="datetime64[ns]")
        date_ord = _to_day_ord_numpy(dts)

    return X, y, ti, dates, features, t2i


def compute_scaler_from_train(X: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xt = X[train_mask]
    if Xt.size == 0:
        raise RuntimeError("No training data for scaler.")
    F = Xt.shape[-1]
    mu = np.nanmean(Xt.reshape(-1, F), axis=0)
    sd = np.nanstd(Xt.reshape(-1, F), axis=0) + 1e-9
    return mu, sd


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, device, epochs=10, lr=1e-3,
                amp: bool = True, weight_decay: float = 1e-3, clip_grad: float = 1.0,
                lr_plateau_patience: int = 1, lr_plateau_factor: float = 0.5, lr_min: float = 1e-5,
                early_stop_patience: int = 9, noise_std: float = 0.02, rank_weight: float = 0.3,
                time_decay_half_life: int = 365, max_train_ordinal: int = None, ema=None,
                loss_type: str = 'huber', huber_delta: float = 1.0, rank_loss: str = 'listwise', listwise_temp: float = 1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=lr_plateau_factor,
                                                       patience=lr_plateau_patience, min_lr=lr_min)
    scaler = GradScaler(enabled=(amp and device.type == "cuda"))
    mse_fn = nn.MSELoss(reduction="none")

    best_val = math.inf
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        n_obs = 0
        pbar = tqdm(
            train_loader, desc=f"Train Epoch {ep}/{epochs}", leave=False)
        for xb, yb, tb, db in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            tb = tb.to(device)
            if noise_std > 0:
                xb = xb + noise_std * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                mu = model(xb, tb)
                base_vec = _huber_loss(
                    mu, yb, delta=huber_delta, reduction='none') if loss_type == 'huber' else mse_fn(mu, yb)
                if time_decay_half_life and max_train_ordinal is not None and db is not None:
                    w = _weights_time_decay(
                        db.cpu().numpy(), max_train_ordinal, time_decay_half_life)
                    base = (base_vec * w.to(device)).mean()
                else:
                    base = base_vec.mean()
                rloss = None
                if rank_loss == 'top1':
                    rloss = _rank_top1_loss(
                        mu, yb, db.cpu().numpy() if db is not None else None)
                elif rank_loss == 'listwise':
                    rloss = _rank_listwise_loss(mu, yb, db.cpu().numpy(
                    ) if db is not None else None, temp=listwise_temp)
                loss = (1.0 - rank_weight) * base + \
                    (rank_weight * rloss if rloss is not None else 0.0)

            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            tr_loss += loss.item() * xb.size(0)
            n_obs += xb.size(0)
        tr_loss /= max(1, n_obs)

        # validation
        val_preds, val_true = [], []
        model.eval()
        with torch.no_grad():
            pbar = tqdm(
                val_loader, desc=f"Valid Epoch {ep}/{epochs}", leave=False)
            for xb, yb, tb, db in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                tb = tb.to(device)
                with autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                    mu = model(xb, tb)
                val_preds.append(mu.detach().cpu().numpy())
                val_true.append(yb.cpu().numpy())

        vp = np.concatenate(val_preds) if val_preds else np.zeros(
            (0,), dtype=np.float32)
        vy = np.concatenate(val_true) if val_true else np.zeros(
            (0,), dtype=np.float32)
        val_rmse = float(np.sqrt(np.mean((vp - vy)**2))
                         ) if vp.size else float("inf")
        val_mae = float(np.mean(np.abs(vp - vy))) if vp.size else float("inf")
        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.5f} | val_rmse={val_rmse:.5f} | val_mae={val_mae:.5f}")

        sched.step(val_rmse)
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            bad = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop_patience and ep > epochs // 2:
                break

    if 'best_state' in locals():
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device) for k, v in best_state.items()})
    return model


def evaluate(model, loader: DataLoader, device, amp=True):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", leave=False)
        for xb, yb, tb, db in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            tb = tb.to(device)
            with autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                mu = model(xb, tb)
            preds.append(mu.detach().cpu().numpy())
            ys.append(yb.cpu().numpy())
    p = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float32)
    y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.float32)
    metrics = {
        "rmse": float(np.sqrt(np.mean((p-y)**2))) if p.size else float("nan"),
        "mae":  float(np.mean(np.abs(p-y))) if p.size else float("nan")
    }
    return p, y, metrics


class EMA:
    def __init__(self, model, decay=0.995):
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)
