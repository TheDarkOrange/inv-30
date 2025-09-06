# S&P 500 — 30-Day Return Prediction (GPU-Optimized LSTM)

**Fast on RTX 2070** with:
- PyTorch **AMP (mixed precision)** on GPU
- **Pinned-memory** DataLoaders (`pin_memory`, `non_blocking=True` transfers)
- **num_workers** + `persistent_workers` + `prefetch_factor`
- Optional **torch.compile** (flag; falls back if unsupported)
- **cudnn.benchmark=True** for fixed-length sequences

**Model**: LSTM → MLP head predicts **30-day log-return**; we use **Adj Close** prices.  
**Backtest**: monthly pick-the-best; converts logret → ROI via `exp(logret)-1`, caps per-trade ROI; compares to SPY.

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## Useful flags
- `--n 10` (default), `--fast 1`
- `--history_years 12`, `--window 60`, `--horizon 30`
- `--epochs 12 --hidden 128 --layers 2 --dropout 0.1`
- `--num_workers 4 --pin_memory 1 --prefetch_factor 4`
- `--amp 1` (default on GPU), `--compile 0`
- `--max_roi_per_trade 0.6` (cap monthly ROI), `--use_adj_close 1`
- `--wandb 1 --wandb_mode offline` (for local logging)

Artifacts:
- `artifacts/model.pt`, `artifacts/metrics.json`
- `artifacts/test_predictions_with_confidence.csv`
- `artifacts/backtest.csv`
