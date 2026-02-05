#!/usr/bin/env python3
"""
Position-Based Model with Product Features

Mathematical Framework:
- P(click | position k, features x) = θ_k × α(x; φ)
- θ_k = k^β₁ (log-linear position effect, β₁ ≤ 0)
- α(x; φ) = attractiveness function of product features

Two models implemented:
1. Minimal: Logistic regression with logit(p) ≈ β₁ log(k) + β₂ log(1+price) + β₃ quality
2. Neural: Lightweight P = θ·α model with MLP for α(price, quality, category)

Reference: Extends Craswell et al. (2008) PBM with feature-based attractiveness.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_R2_DIR = BASE_DIR.parent / "data_r2"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "11_feature_click_model_results.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# DATA PREPARATION
# =============================================================================
def load_and_merge_features(log_fn):
    """Load session_items and merge with catalog features."""
    log_fn("Loading data...")

    items = pd.read_parquet(DATA_DIR / "session_items.parquet")
    log_fn(f"  Session items: {len(items):,} rows")

    catalog = pd.read_parquet(DATA_R2_DIR / "catalog_r2.parquet")
    log_fn(f"  Catalog: {len(catalog):,} products")

    items = items.merge(
        catalog[['PRODUCT_ID', 'CATALOG_PRICE', 'CATEGORIES']],
        left_on='product_id', right_on='PRODUCT_ID', how='left'
    )

    has_price = items['CATALOG_PRICE'].notna().mean()
    has_cat = items['CATEGORIES'].notna().mean()

    log_fn(f"  Price coverage: {has_price*100:.1f}%")
    log_fn(f"  Categories coverage: {has_cat*100:.1f}%")

    return items


def extract_primary_category(categories_str):
    """Extract primary category from categories array string."""
    if pd.isna(categories_str):
        return 'unknown'
    try:
        cats = json.loads(categories_str) if isinstance(categories_str, str) else categories_str
        for cat in cats:
            if isinstance(cat, str) and cat.startswith('category#'):
                return cat.replace('category#', '')
        return 'unknown'
    except:
        return 'unknown'


def prepare_features(items, log_fn):
    """Prepare numeric features."""
    log_fn("Preparing features...")

    items['log_price'] = np.log1p(items['CATALOG_PRICE'].fillna(0))
    items['quality_filled'] = items['quality'].fillna(0)
    items['primary_category'] = items['CATEGORIES'].apply(extract_primary_category)

    cat_counts = items['primary_category'].value_counts()
    top_categories = cat_counts.head(49).index.tolist()
    category_to_idx = {cat: idx for idx, cat in enumerate(top_categories)}
    category_to_idx['other'] = len(top_categories)

    items['category_idx'] = items['primary_category'].apply(
        lambda x: category_to_idx.get(x, category_to_idx['other'])
    )

    log_fn(f"  Categories encoded: {len(category_to_idx)}")
    return items, category_to_idx


# =============================================================================
# MODEL 1: MINIMAL LOGISTIC REGRESSION
# =============================================================================
class MinimalClickModel:
    """logit(p) = β₀ + β₁ log(k) + β₂ log(1+price) + β₃ quality"""

    def __init__(self):
        self.model = None

    def prepare_features(self, df):
        return np.column_stack([
            np.log(df['position'].values),
            df['log_price'].values,
            df['quality_filled'].values
        ])

    def fit(self, X, y):
        self.model = LogisticRegression(penalty=None, max_iter=1000)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self):
        return {
            'intercept': self.model.intercept_[0],
            'log_position': self.model.coef_[0][0],
            'log_price': self.model.coef_[0][1],
            'quality': self.model.coef_[0][2]
        }


# =============================================================================
# MODEL 2: NEURAL FEATURE MODEL (LIGHTWEIGHT)
# =============================================================================
class ClickDataset(Dataset):
    def __init__(self, positions, log_prices, qualities, category_ids, clicked):
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.log_prices = torch.tensor(log_prices, dtype=torch.float32)
        self.qualities = torch.tensor(qualities, dtype=torch.float32)
        self.category_ids = torch.tensor(category_ids, dtype=torch.long)
        self.clicked = torch.tensor(clicked, dtype=torch.float32)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            'position': self.positions[idx],
            'log_price': self.log_prices[idx],
            'quality': self.qualities[idx],
            'category_id': self.category_ids[idx],
            'clicked': self.clicked[idx]
        }


class FeatureClickModel(nn.Module):
    """
    P(click) = θ_k · α(x; φ)
    θ_k = k^β₁
    α(x) = σ(MLP([price; quality; category_embed]))
    """

    def __init__(self, cat_dim=16, hidden_dim=32, n_categories=50):
        super().__init__()
        self.beta1 = nn.Parameter(torch.tensor(-0.05))
        self.category_embed = nn.Embedding(n_categories, cat_dim)

        input_dim = 1 + 1 + cat_dim  # price + quality + category
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, position, log_price, quality, category_id):
        beta1_clamped = torch.clamp(self.beta1, max=0.0)
        theta = torch.pow(position.float() + 1e-6, beta1_clamped)

        cat_embed = self.category_embed(category_id)
        z = torch.cat([log_price.unsqueeze(1), quality.unsqueeze(1), cat_embed], dim=1)

        alpha = torch.sigmoid(self.mlp(z)).squeeze(1)
        p_click = torch.clamp(theta * alpha, min=1e-7, max=1-1e-7)

        return p_click, theta, alpha

    def loss(self, p_click, y):
        eps = 1e-7
        p_click = torch.clamp(p_click, min=eps, max=1-eps)
        return -torch.mean(y * torch.log(p_click) + (1-y) * torch.log(1 - p_click))


def train_neural_model(model, train_loader, val_loader, epochs=20, lr=1e-3, log_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            p_click, _, _ = model(
                batch['position'], batch['log_price'],
                batch['quality'], batch['category_id']
            )
            loss = model.loss(p_click, batch['clicked'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if log_fn and (epoch % 5 == 0 or epoch == epochs - 1):
            log_fn(f"    Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}, β₁={model.beta1.item():.4f}")

    return model


def evaluate_model(model, data_loader, return_predictions=False):
    model.eval()
    all_preds, all_labels, all_theta, all_alpha = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            p_click, theta, alpha = model(
                batch['position'], batch['log_price'],
                batch['quality'], batch['category_id']
            )
            all_preds.extend(p_click.numpy())
            all_labels.extend(batch['clicked'].numpy())
            all_theta.extend(theta.numpy())
            all_alpha.extend(alpha.numpy())

    preds = np.clip(np.array(all_preds), 1e-7, 1-1e-7)
    labels = np.array(all_labels)

    metrics = {
        'log_loss': log_loss(labels, preds),
        'auc': roc_auc_score(labels, preds),
        'brier': brier_score_loss(labels, preds)
    }

    if return_predictions:
        return metrics, preds, labels, np.array(all_theta), np.array(all_alpha)
    return metrics


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("POSITION-BASED MODEL WITH PRODUCT FEATURES", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  P(click | position k, features x) = θ_k × α(x; φ)", f)
        log("  θ_k = k^β₁ — log-linear position effect (β₁ ≤ 0)", f)
        log("  α(x; φ) — product attractiveness from features", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Data Loading
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)

        log_fn = lambda msg: log(msg, f)
        items = load_and_merge_features(log_fn)
        items, category_to_idx = prepare_features(items, log_fn)

        log("", f)
        log(f"Total impressions: {len(items):,}", f)
        log(f"Total clicks: {items['clicked'].sum():,}", f)
        log(f"Overall CTR: {items['clicked'].mean()*100:.3f}%", f)
        log("", f)

        # Position distribution
        log("Position distribution:", f)
        pos_dist = items.groupby('position').agg({'clicked': ['count', 'sum', 'mean']}).head(10)
        pos_dist.columns = ['n', 'clicks', 'ctr']
        for pos in pos_dist.index:
            row = pos_dist.loc[pos]
            log(f"  Position {pos}: n={int(row['n']):,}, CTR={row['ctr']*100:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Train/Test Split
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: TRAIN/TEST SPLIT", f)
        log("-" * 40, f)

        unique_auctions = items['auction_id'].unique()
        train_auctions, test_auctions = train_test_split(unique_auctions, test_size=0.2, random_state=42)
        train_auctions, val_auctions = train_test_split(train_auctions, test_size=0.125, random_state=42)

        train_items = items[items['auction_id'].isin(train_auctions)]
        val_items = items[items['auction_id'].isin(val_auctions)]
        test_items = items[items['auction_id'].isin(test_auctions)]

        log(f"Train: {len(train_items):,} ({train_items['clicked'].mean()*100:.2f}% CTR)", f)
        log(f"Val:   {len(val_items):,} ({val_items['clicked'].mean()*100:.2f}% CTR)", f)
        log(f"Test:  {len(test_items):,} ({test_items['clicked'].mean()*100:.2f}% CTR)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Minimal Logistic Regression
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: MINIMAL LOGISTIC REGRESSION", f)
        log("-" * 40, f)

        log("MODEL: logit(p) = β₀ + β₁ log(k) + β₂ log(1+price) + β₃ quality", f)
        log("", f)

        minimal_model = MinimalClickModel()
        X_train = minimal_model.prepare_features(train_items)
        X_test = minimal_model.prepare_features(test_items)
        y_train = train_items['clicked'].values
        y_test = test_items['clicked'].values

        minimal_model.fit(X_train, y_train)
        coefs = minimal_model.get_coefficients()

        log("Coefficients:", f)
        log(f"  β₀ (intercept):    {coefs['intercept']:.4f}", f)
        log(f"  β₁ (log position): {coefs['log_position']:.4f}", f)
        log(f"  β₂ (log price):    {coefs['log_price']:.4f}", f)
        log(f"  β₃ (quality):      {coefs['quality']:.4f}", f)
        log("", f)

        log("Interpretation:", f)
        if coefs['log_position'] < 0:
            log(f"  Position: Higher positions reduce click probability", f)
        else:
            log(f"  Position: Weak or no position decay", f)
        log("", f)

        pred_test = minimal_model.predict_proba(X_test)
        minimal_metrics = {
            'log_loss': log_loss(y_test, pred_test),
            'auc': roc_auc_score(y_test, pred_test),
            'brier': brier_score_loss(y_test, pred_test)
        }

        log("Test Performance:", f)
        log(f"  Log Loss: {minimal_metrics['log_loss']:.4f}", f)
        log(f"  AUC:      {minimal_metrics['auc']:.4f}", f)
        log(f"  Brier:    {minimal_metrics['brier']:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Neural Feature Model
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: NEURAL FEATURE MODEL", f)
        log("-" * 40, f)

        log("MODEL: P(click) = θ_k × α(x; φ)", f)
        log("  θ_k = k^β₁", f)
        log("  α(x) = σ(MLP([price; quality; category]))", f)
        log("", f)

        train_dataset = ClickDataset(
            train_items['position'].values, train_items['log_price'].values,
            train_items['quality_filled'].values, train_items['category_idx'].values,
            train_items['clicked'].values
        )
        val_dataset = ClickDataset(
            val_items['position'].values, val_items['log_price'].values,
            val_items['quality_filled'].values, val_items['category_idx'].values,
            val_items['clicked'].values
        )
        test_dataset = ClickDataset(
            test_items['position'].values, test_items['log_price'].values,
            test_items['quality_filled'].values, test_items['category_idx'].values,
            test_items['clicked'].values
        )

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        neural_model = FeatureClickModel(n_categories=len(category_to_idx))

        log("Training...", f)
        neural_model = train_neural_model(neural_model, train_loader, val_loader, epochs=20, log_fn=log_fn)
        log("", f)

        beta1_learned = neural_model.beta1.item()
        log(f"Learned position elasticity β₁ = {beta1_learned:.4f}", f)
        log(f"  Position 5: θ_5 = {5**beta1_learned:.4f}", f)
        log(f"  Position 10: θ_10 = {10**beta1_learned:.4f}", f)
        log("", f)

        test_metrics, test_preds, test_labels, test_theta, test_alpha = evaluate_model(
            neural_model, test_loader, return_predictions=True
        )

        log("Test Performance:", f)
        log(f"  Log Loss: {test_metrics['log_loss']:.4f}", f)
        log(f"  AUC:      {test_metrics['auc']:.4f}", f)
        log(f"  Brier:    {test_metrics['brier']:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Position Effect Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: POSITION EFFECT COMPARISON", f)
        log("-" * 40, f)

        log("Position effects θ_k = k^β₁:", f)
        log(f"  {'Position':<10} {'θ_k (Neural)':<15}", f)
        log(f"  {'-'*10} {'-'*15}", f)
        for k in range(1, 11):
            log(f"  {k:<10} {k**beta1_learned:<15.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Model Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: MODEL COMPARISON", f)
        log("-" * 40, f)

        pbm_baseline_ll = 0.1259
        pbm_baseline_auc = 0.5669

        log("Test set comparison:", f)
        log(f"  {'Model':<25} {'Log Loss':<12} {'AUC':<12}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*12}", f)
        log(f"  {'PBM Baseline (EM)':<25} {pbm_baseline_ll:<12.4f} {pbm_baseline_auc:<12.4f}", f)
        log(f"  {'Minimal (Logistic)':<25} {minimal_metrics['log_loss']:<12.4f} {minimal_metrics['auc']:<12.4f}", f)
        log(f"  {'Neural Feature Model':<25} {test_metrics['log_loss']:<12.4f} {test_metrics['auc']:<12.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Attractiveness Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: ATTRACTIVENESS ANALYSIS", f)
        log("-" * 40, f)

        log("Attractiveness α(x) statistics:", f)
        log(f"  Mean: {test_alpha.mean():.4f}", f)
        log(f"  Std:  {test_alpha.std():.4f}", f)
        log(f"  Min:  {test_alpha.min():.4f}", f)
        log(f"  Max:  {test_alpha.max():.4f}", f)
        log("", f)

        test_items_copy = test_items.copy()
        test_items_copy['pred_alpha'] = test_alpha
        test_items_copy['pred_theta'] = test_theta

        log("Mean α by quality quintile:", f)
        test_items_copy['q_quintile'] = pd.qcut(
            test_items_copy['quality_filled'].clip(lower=1e-6), q=5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop'
        )
        for q, alpha in test_items_copy.groupby('q_quintile')['pred_alpha'].mean().items():
            log(f"  {q}: α = {alpha:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("POSITION EFFECTS:", f)
        log(f"  Model: θ_k = k^β₁ with β₁ = {beta1_learned:.4f}", f)
        log(f"  Position 1: θ_1 = 1.000", f)
        log(f"  Position 5: θ_5 = {5**beta1_learned:.4f}", f)
        log(f"  Position 10: θ_10 = {10**beta1_learned:.4f}", f)
        log("", f)

        log("KEY FINDINGS:", f)
        if beta1_learned < -0.05:
            log("  1. Position decay is significant", f)
        else:
            log("  1. Position decay is weak or absent", f)

        best_model = "Neural" if test_metrics['auc'] > minimal_metrics['auc'] else "Minimal"
        log(f"  2. Best model: {best_model}", f)
        log(f"  3. Neural AUC: {test_metrics['auc']:.4f} vs Minimal: {minimal_metrics['auc']:.4f}", f)
        log("", f)

        log("=" * 80, f)
        log("COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
