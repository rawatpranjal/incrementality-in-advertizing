"""
Causal Transformer Model for Ad Effectiveness

This module implements a Transformer-based model with dual heads for estimating:
1. α(X): Baseline purchase propensity
2. β(X): Causal lift from treatment (ad click)

The model processes sequences of (event_type, item_id, timedelta) tuples
and predicts Y = σ(α(X) + β(X) * T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer models.
    Adds position information to embeddings using sine and cosine functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Add 1 for the treatment token that gets appended
        max_len = max_len + 1

        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0), :]


class CausalTransformer(nn.Module):
    """
    Transformer-based model for causal inference in ad effectiveness.

    Architecture:
    1. Event embeddings (type + item + time)
    2. Transformer encoder layers
    3. Dual output heads for α(X) and β(X)

    Args:
        item_vocab_size: Number of unique items/products
        event_vocab_size: Number of event types (auction, impression, click, purchase)
        d_model: Hidden dimension size
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        max_seq_length: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        item_vocab_size: int,
        event_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 500
    ):
        super().__init__()

        self.d_model = d_model

        # Embedding layers
        self.item_embedding = nn.Embedding(item_vocab_size, d_model, padding_idx=0)
        self.event_embedding = nn.Embedding(event_vocab_size, d_model)

        # Time delta projection
        self.time_projection = nn.Linear(1, d_model)

        # Layer normalization for combined embeddings
        self.embedding_norm = nn.LayerNorm(d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pooling strategy: use [CLS] token (first position)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Causal heads
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.beta_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def create_padding_mask(self, sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Create attention mask for padded sequences.

        Args:
            sequence_lengths: Actual length of each sequence in batch
            max_length: Maximum length in the batch

        Returns:
            Boolean mask where True indicates positions to ignore
        """
        batch_size = sequence_lengths.size(0)
        mask = torch.arange(max_length, device=sequence_lengths.device).expand(
            batch_size, max_length
        ) >= sequence_lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        event_types: torch.Tensor,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        sequence_lengths: torch.Tensor,
        treatments: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Causal Transformer.

        Args:
            event_types: (batch_size, seq_len) Event type IDs
            item_ids: (batch_size, seq_len) Item/product IDs
            time_deltas: (batch_size, seq_len) Time deltas in minutes
            sequence_lengths: (batch_size,) Actual length of each sequence
            treatments: (batch_size,) Binary treatment indicators (optional)

        Returns:
            logits: (batch_size,) Final prediction logits
            alpha: (batch_size,) Baseline propensity α(X)
            beta: (batch_size,) Causal lift β(X)
        """
        batch_size = event_types.size(0)
        seq_len = event_types.size(1)

        # Get embeddings
        event_emb = self.event_embedding(event_types)  # (batch, seq, d_model)
        item_emb = self.item_embedding(item_ids)  # (batch, seq, d_model)

        # Project time deltas
        time_emb = self.time_projection(time_deltas.unsqueeze(-1))  # (batch, seq, d_model)

        # Combine embeddings
        embeddings = event_emb + item_emb + time_emb
        embeddings = self.embedding_norm(embeddings)

        # Add [CLS] token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Update sequence lengths for [CLS] token
        sequence_lengths = sequence_lengths + 1

        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (seq+1, batch, d_model)
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch, seq+1, d_model)

        # Create padding mask
        padding_mask = self.create_padding_mask(sequence_lengths, embeddings.size(1))

        # Pass through transformer
        encoded = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=padding_mask
        )

        # Extract [CLS] token representation
        cls_output = encoded[:, 0, :]  # (batch, d_model)

        # Compute causal outputs
        alpha = self.alpha_head(cls_output).squeeze(-1)  # (batch,)
        beta = self.beta_head(cls_output).squeeze(-1)  # (batch,)

        # Compute final logits if treatments provided
        if treatments is not None:
            logits = alpha + beta * treatments.float()
        else:
            # Return separate components for analysis
            logits = alpha  # Default to baseline

        return logits, alpha, beta

    def predict_outcome_probability(
        self,
        event_types: torch.Tensor,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        sequence_lengths: torch.Tensor,
        treatments: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict purchase probability for given sequences and treatments.

        Returns:
            Probabilities (batch_size,) after sigmoid activation
        """
        logits, _, _ = self.forward(
            event_types, item_ids, time_deltas, sequence_lengths, treatments
        )
        return torch.sigmoid(logits)

    def estimate_treatment_effect(
        self,
        event_types: torch.Tensor,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate individual treatment effects for sequences.

        Returns:
            alpha: Baseline purchase probability
            beta: Treatment effect (lift from click)
        """
        with torch.no_grad():
            _, alpha, beta = self.forward(
                event_types, item_ids, time_deltas, sequence_lengths
            )
            # Convert to probabilities
            alpha_prob = torch.sigmoid(alpha)
            # Beta represents the additive effect in logit space
            # To get probability lift, we need to be careful
            treated_prob = torch.sigmoid(alpha + beta)
            beta_lift = treated_prob - alpha_prob

        return alpha_prob, beta_lift