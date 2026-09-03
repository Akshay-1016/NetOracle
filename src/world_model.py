"""
World Model Core: Temporal Transformer with Multi-Task Forecasting Heads
Learns Transition Dynamics P(S_{t+1} | S_t, ..., S_{t-L})
"""
import torch
import torch.nn as nn
import math
from typing import Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class NetworkWorldModel(nn.Module):
    def __init__(self, input_dim: int, state_dim: int = 64, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 3, forecast_horizon: int = 5,
                 num_mitre_stages: int = 7, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.forecast_horizon = forecast_horizon
        self.num_mitre_stages = num_mitre_stages

        # State Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU()
        )

        # Temporal Transition Transformer
        self.pos_encoder = PositionalEncoding(state_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Multi-task Rollout Heads
        self.state_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim * forecast_horizon)
        )
        
        self.attack_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, forecast_horizon),
            nn.Sigmoid()
        )
        
        self.stage_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_mitre_stages * forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        z = self.encoder(x_flat).reshape(batch_size, seq_len, self.state_dim)
        z = self.pos_encoder(z)

        # Causal Attention Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        context = self.transformer(z, mask=mask)
        last_step = context[:, -1, :]

        # Predictions
        pred_states = self.state_head(last_step).reshape(batch_size, self.forecast_horizon, self.input_dim)
        pred_attacks = self.attack_head(last_step)
        pred_stages = self.stage_head(last_step).reshape(batch_size, self.forecast_horizon, self.num_mitre_stages)

        return {
            'predicted_states': pred_states,
            'attack_probs': pred_attacks,
            'mitre_logits': pred_stages
        }
