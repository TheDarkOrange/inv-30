import torch
import torch.nn as nn


class RNNRegressor(nn.Module):
    def __init__(self, rnn_type: str, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, n_tickers: int, ticker_embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=n_tickers, embedding_dim=ticker_embed_dim)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features + ticker_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, ticker_ids):
        # x: [B, T, F]; ticker_ids: [B]
        emb = self.embed(ticker_ids)            # [B, E]
        emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, E]
        xcat = torch.cat([x, emb_seq], dim=-1)  # [B, T, F+E]
        out, _ = self.rnn(xcat)
        last = out[:, -1, :]                    # [B, H]
        y = self.head(last).squeeze(-1)         # [B]
        return y
