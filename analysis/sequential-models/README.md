# Sequential Models (RNN/LSTM)

## Overview

Sequential modeling using recurrent neural networks (RNN/LSTM) to capture temporal dependencies in user-product interaction sequences. This approach models the ad funnel as a sequence of events and learns representations that capture the dynamic path from impressions to clicks to purchases.

## Data Requirements

- **Unit of analysis:** User-product interaction sequences
- **Input tables:** IMPRESSIONS, CLICKS, PURCHASES, AUCTIONS_USERS
- **Sample/filters:** Users with sufficient interaction history; sequences ordered by timestamp
- **Features:** Event type, product embeddings, time gaps, auction context

## Pipeline

1. `01_data_pull.ipynb` — Extract raw interaction data from warehouse
2. `02_data_processing.ipynb` — Sequence construction and feature engineering
3. `03_model_training.ipynb` — Train RNN/LSTM models
4. `04_causal_analysis.ipynb` — Interpret learned representations for causal inference

## Model Specification

**LSTM Sequence Model:**
```
h_t = LSTM(x_t, h_{t-1})
ŷ = σ(W·h_T + b)
```
where:
- x_t = input features at time t (event type, product, context)
- h_t = hidden state capturing sequence history
- ŷ = predicted outcome (purchase probability)

**Loss Function:**
```
L = -Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```
Binary cross-entropy for purchase prediction

**Causal Application:**
- Compare predicted outcomes with/without ad exposure events in sequence
- Counterfactual: Remove ad events from sequence, re-predict

**Interpretation:**
- Hidden states learn temporal patterns in user journeys
- Attention weights (if used) reveal which events drive predictions
- Counterfactual comparison estimates ad contribution

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | LSTM model architecture definition |
| `src/dataloader.py` | Sequence data loading and batching |
| `01_data_pull.ipynb` | Data extraction |
| `02_data_processing.ipynb` | Sequence construction |
| `03_model_training.ipynb` | Model training and validation |
| `04_causal_analysis.ipynb` | Causal interpretation of results |

## Outputs

- Trained LSTM model weights
- Sequence embeddings
- Purchase prediction accuracy metrics
- Counterfactual effect estimates
- Feature importance from sequence analysis

## Connections

- Relates to `deep-learning/` for neural network approaches to treatment effects
- Complements `shopping-sessions/` which uses session-based (not sequence) aggregation
- Captures dynamics that `panel/` fixed effects cannot model
