from dataclasses import dataclass
import argparse


@dataclass
class Config:
    # data
    n: int = 40
    history_years: int = 10
    window: int = 120
    horizon: int = 30
    use_adj_close: int = 1
    benchmark_symbol: str = "SPY"

    # train
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    amp: int = 1
    compile: int = 0
    seed: int = 42

    # model arch
    rnn_type: str = "gru"        # gru|lstm
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2

    # ensemble
    m: int = 1                   # number of ensemble members

    # selection / confidence
    k: int = 2                   # pick top-k each decision
    lambda_conf: float = 1.0     # λ for mean - λ*std
    weighting: str = "equal"     # equal|softmax
    softmax_temp: float = 1.0    # temperature if weighting=softmax
    hold_if_uncertain: int = 1   # 1=hold SPY when top score<=0; 0=skip trade

    # loss
    loss: str = "huber"          # mse|huber
    huber_delta: float = 1.0
    rank_loss: str = "listwise"  # none|top1|listwise
    rank_weight: float = 0.3
    listwise_temp: float = 1.0

    # features
    extra_features: int = 1      # add momentum/ATR/drawdown/vol_z

    # blender
    blend_ridge: int = 0
    blend_alpha: float = 1.0

    t_recent_months: int = 12


def parse_args() -> 'Config':
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--history_years", type=int, default=10)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--use_adj_close", type=int, default=1)
    p.add_argument("--benchmark_symbol", type=str, default="SPY")
    # train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--rnn_type", type=str, default="gru",
                   choices=["gru", "lstm"])
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    # ensemble
    p.add_argument("--m", type=int, default=1)
    # selection / confidence
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lambda_conf", type=float, default=1.0)
    p.add_argument("--weighting", type=str, default="equal",
                   choices=["equal", "softmax"])
    p.add_argument("--softmax_temp", type=float, default=1.0)
    p.add_argument("--hold_if_uncertain", type=int, default=1)
    # loss
    p.add_argument("--loss", type=str, default="huber",
                   choices=["mse", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--rank_loss", type=str, default="listwise",
                   choices=["none", "top1", "listwise"])
    p.add_argument("--rank_weight", type=float, default=0.3)
    p.add_argument("--listwise_temp", type=float, default=1.0)
    # features
    p.add_argument("--extra_features", type=int, default=1)
    # blender
    p.add_argument("--blend_ridge", type=int, default=0)
    p.add_argument("--blend_alpha", type=float, default=1.0)

    p.add_argument("--t_recent_months", type=int, default=12)

    a = p.parse_args()
    return Config(
        n=a.n, history_years=a.history_years, window=a.window, horizon=a.horizon,
        use_adj_close=a.use_adj_close, benchmark_symbol=a.benchmark_symbol,
        batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, amp=a.amp, compile=a.compile, seed=a.seed,
        rnn_type=a.rnn_type, hidden_size=a.hidden_size, num_layers=a.num_layers, dropout=a.dropout,
        m=a.m, k=a.k, lambda_conf=a.lambda_conf, weighting=a.weighting, softmax_temp=a.softmax_temp,
        hold_if_uncertain=a.hold_if_uncertain, loss=a.loss, huber_delta=a.huber_delta,
        rank_loss=a.rank_loss, rank_weight=a.rank_weight, listwise_temp=a.listwise_temp,
        extra_features=a.extra_features, blend_ridge=a.blend_ridge, blend_alpha=a.blend_alpha,
    )
