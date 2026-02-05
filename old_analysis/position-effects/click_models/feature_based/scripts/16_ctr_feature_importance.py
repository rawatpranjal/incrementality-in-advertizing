#!/usr/bin/env python3
"""
CTR Prediction Model with Feature Importances

Objective:
Build a comprehensive CTR prediction model using all available features,
avoiding target leakage, and report feature importances.

Unit of Analysis: Individual item impressions within sessions
Target Variable: clicked (binary)

Features:
- Position features: position, log_position, relative_position, row, col, is_top_row, is_first_col
- Product features: quality, log_quality, bid, log_bid
- Relative features: relative_quality, relative_bid, quality_rank, bid_rank
- Session context: n_items, log_n_items, placement one-hot

Features EXCLUDED (leakage):
- n_clicks: session-level aggregate of target

Model: XGBoost classifier (sklearn GradientBoosting fallback)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "16_ctr_feature_importance.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(items_df, grid_width=2):
    """
    Create feature matrix for CTR prediction.

    INCLUDED (no leakage):
    - Position features: position, log_position, relative_position, row, col, is_top_row, is_first_col
    - Product features: quality, log_quality, bid, log_bid
    - Relative features (within-session, no target): relative_quality, relative_bid, quality_rank, bid_rank
    - Session context: n_items, log_n_items, placement one-hot

    EXCLUDED (leakage):
    - n_clicks: aggregate of target variable
    """
    df = items_df.copy()

    # Position features
    df['log_position'] = np.log(df['position'])
    df['relative_position'] = df['position'] / df['n_items']
    df['row'] = (df['position'] - 1) // grid_width
    df['col'] = (df['position'] - 1) % grid_width
    df['is_top_row'] = (df['row'] == 0).astype(int)
    df['is_first_col'] = (df['col'] == 0).astype(int)

    # Product features
    df['quality_filled'] = df['quality'].fillna(df['quality'].median())
    df['log_quality'] = np.log(df['quality_filled'] + 1e-6)
    df['log_bid'] = np.log1p(df['bid'])

    # Relative features within session (no target leakage)
    session_avg_quality = df.groupby('auction_id')['quality_filled'].transform('mean')
    df['relative_quality'] = df['quality_filled'] / (session_avg_quality + 1e-6)

    session_avg_bid = df.groupby('auction_id')['bid'].transform('mean')
    df['relative_bid'] = df['bid'] / (session_avg_bid + 1e-6)

    # Rank within session (ascending rank = 1 is best)
    df['quality_rank'] = df.groupby('auction_id')['quality_filled'].rank(ascending=False)
    df['bid_rank'] = df.groupby('auction_id')['bid'].rank(ascending=False)

    # Session context
    df['log_n_items'] = np.log(df['n_items'])

    # Placement one-hot encoding
    placement_dummies = pd.get_dummies(df['placement'], prefix='placement')
    df = pd.concat([df, placement_dummies], axis=1)

    return df


def get_feature_columns(df):
    """Get list of feature columns (excluding leakage features)."""

    # Position features
    position_features = [
        'position',
        'log_position',
        'relative_position',
        'row',
        'col',
        'is_top_row',
        'is_first_col'
    ]

    # Product features
    product_features = [
        'quality_filled',
        'log_quality',
        'bid',
        'log_bid'
    ]

    # Relative features (within-session, no target)
    relative_features = [
        'relative_quality',
        'relative_bid',
        'quality_rank',
        'bid_rank'
    ]

    # Session context
    context_features = [
        'n_items',
        'log_n_items'
    ]

    # Placement one-hot
    placement_cols = [c for c in df.columns if c.startswith('placement_')]

    return position_features + product_features + relative_features + context_features + placement_cols


def categorize_features(feature_cols):
    """Categorize features into groups for analysis."""
    categories = {
        'position': [],
        'product_quality': [],
        'product_bid': [],
        'session_context': []
    }

    for feat in feature_cols:
        if any(x in feat.lower() for x in ['position', 'row', 'col', 'top_row', 'first_col']):
            categories['position'].append(feat)
        elif any(x in feat.lower() for x in ['quality']):
            categories['product_quality'].append(feat)
        elif any(x in feat.lower() for x in ['bid']):
            categories['product_bid'].append(feat)
        else:
            categories['session_context'].append(feat)

    return categories


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Compute evaluation metrics."""
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

    return {
        'model': model_name,
        'log_loss': log_loss(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'n_samples': len(y_true)
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("CTR PREDICTION MODEL WITH FEATURE IMPORTANCES", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Data Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        items_df = pd.read_parquet(DATA_DIR / "session_items.parquet")

        log(f"Data source: {DATA_DIR / 'session_items.parquet'}", f)
        log(f"Total rows: {len(items_df):,}", f)
        log(f"Total clicks: {items_df['clicked'].sum():,}", f)
        log(f"Overall CTR: {items_df['clicked'].mean()*100:.2f}%", f)
        log(f"Unique auctions: {items_df['auction_id'].nunique():,}", f)
        log(f"Unique products: {items_df['product_id'].nunique():,}", f)
        log("", f)

        log("Raw column statistics:", f)
        for col in items_df.columns:
            if items_df[col].dtype in ['int64', 'float64']:
                log(f"  {col:<15}: min={items_df[col].min():.4f}, max={items_df[col].max():.4f}, mean={items_df[col].mean():.4f}", f)
            else:
                log(f"  {col:<15}: {items_df[col].nunique()} unique values", f)
        log("", f)

        log("Placements distribution:", f)
        for placement, count in items_df['placement'].value_counts().items():
            pct = count / len(items_df) * 100
            log(f"  Placement {placement}: {count:,} ({pct:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Feature Matrix
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: FEATURE MATRIX", f)
        log("-" * 40, f)
        log("", f)

        log("Creating features...", f)
        items_df = create_features(items_df, grid_width=2)
        feature_cols = get_feature_columns(items_df)

        log(f"Total features: {len(feature_cols)}", f)
        log("", f)

        log("FEATURES INCLUDED:", f)
        for feat in feature_cols:
            if feat in items_df.columns:
                log(f"  - {feat:<20}: [{items_df[feat].min():.4f}, {items_df[feat].max():.4f}], mean={items_df[feat].mean():.4f}", f)
        log("", f)

        log("FEATURES EXCLUDED (leakage prevention):", f)
        log("  - n_clicks: session-level aggregate of target variable", f)
        log("  - clicked: this IS the target variable", f)
        log("", f)

        # Verify n_clicks is NOT in feature list
        assert 'n_clicks' not in feature_cols, "LEAKAGE: n_clicks should not be in features!"
        assert 'clicked' not in feature_cols, "LEAKAGE: clicked should not be in features!"
        log("Leakage check PASSED: n_clicks and clicked are not in feature list.", f)
        log("", f)

        # Feature categorization
        categories = categorize_features(feature_cols)
        log("Feature categories:", f)
        for cat, feats in categories.items():
            log(f"  {cat}: {feats}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Train/Test Split
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: TRAIN/TEST SPLIT", f)
        log("-" * 40, f)
        log("", f)

        log("Splitting by auction_id (80/20)...", f)
        unique_auctions = items_df['auction_id'].unique()
        train_auctions, test_auctions = train_test_split(
            unique_auctions, test_size=0.2, random_state=42
        )

        train_df = items_df[items_df['auction_id'].isin(train_auctions)]
        test_df = items_df[items_df['auction_id'].isin(test_auctions)]

        log(f"Train set: {len(train_df):,} items, {train_df['clicked'].sum():,} clicks ({train_df['clicked'].mean()*100:.2f}% CTR)", f)
        log(f"Test set:  {len(test_df):,} items, {test_df['clicked'].sum():,} clicks ({test_df['clicked'].mean()*100:.2f}% CTR)", f)
        log("", f)

        # Prepare feature matrices
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df['clicked'].values
        y_test = test_df['clicked'].values

        log(f"Feature matrix shape: {X_train.shape}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Model Training
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: MODEL TRAINING", f)
        log("-" * 40, f)
        log("", f)

        model_type = "XGBoost" if HAS_XGBOOST else "sklearn GradientBoosting"
        log(f"Model: {model_type}", f)
        log("Hyperparameters:", f)
        log("  n_estimators: 200", f)
        log("  max_depth: 5", f)
        log("  learning_rate: 0.1", f)
        log("", f)

        log("Training model...", f)
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        model.fit(X_train, y_train)
        log("Training complete.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Model Performance
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: MODEL PERFORMANCE", f)
        log("-" * 40, f)
        log("", f)

        # Train predictions
        train_pred = model.predict_proba(X_train)[:, 1]
        train_metrics = evaluate_predictions(y_train, train_pred, "Train")

        # Test predictions
        test_pred = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, test_pred, "Test")

        log("TRAIN SET METRICS:", f)
        log(f"  Log Loss: {train_metrics['log_loss']:.6f}", f)
        log(f"  AUC:      {train_metrics['auc']:.6f}", f)
        log(f"  Brier:    {train_metrics['brier']:.6f}", f)
        log("", f)

        log("TEST SET METRICS:", f)
        log(f"  Log Loss: {test_metrics['log_loss']:.6f}", f)
        log(f"  AUC:      {test_metrics['auc']:.6f}", f)
        log(f"  Brier:    {test_metrics['brier']:.6f}", f)
        log("", f)

        # Verify AUC > 0.5
        assert test_metrics['auc'] > 0.5, f"AUC should be > 0.5 (got {test_metrics['auc']:.4f})"
        log(f"AUC check PASSED: {test_metrics['auc']:.4f} > 0.5 (better than random)", f)
        log("", f)

        # Overfitting check
        overfit_ratio = train_metrics['auc'] / test_metrics['auc']
        log(f"Overfitting check: Train AUC / Test AUC = {overfit_ratio:.4f}", f)
        if overfit_ratio > 1.1:
            log("  Warning: Possible overfitting (ratio > 1.1)", f)
        else:
            log("  No significant overfitting detected.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Feature Importances
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: FEATURE IMPORTANCES", f)
        log("-" * 40, f)
        log("", f)

        importance = model.feature_importances_
        importance_dict = dict(zip(feature_cols, importance))
        total_importance = sum(importance)

        log("Feature importances (sorted by importance):", f)
        log(f"  {'Feature':<25} {'Importance':<12} {'Percentage':<10}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*10}", f)

        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance:
            pct = imp / total_importance * 100
            log(f"  {feat:<25} {imp:<12.6f} {pct:>6.2f}%", f)
        log("", f)

        # Verify importances sum to ~100%
        total_pct = sum(imp / total_importance * 100 for _, imp in sorted_importance)
        log(f"Total percentage: {total_pct:.2f}% (should be ~100%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Feature Category Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: FEATURE CATEGORY ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("Aggregate importance by feature category:", f)
        log("", f)

        category_importance = {}
        for cat, feats in categories.items():
            cat_imp = sum(importance_dict.get(feat, 0) for feat in feats)
            cat_pct = cat_imp / total_importance * 100
            category_importance[cat] = {'importance': cat_imp, 'pct': cat_pct, 'n_features': len(feats)}

        log(f"  {'Category':<20} {'Features':<10} {'Total Importance':<18} {'Percentage':<12}", f)
        log(f"  {'-'*20} {'-'*10} {'-'*18} {'-'*12}", f)
        for cat, vals in sorted(category_importance.items(), key=lambda x: x[1]['pct'], reverse=True):
            log(f"  {cat:<20} {vals['n_features']:<10} {vals['importance']:<18.6f} {vals['pct']:>8.2f}%", f)
        log("", f)

        # Determine dominant category
        dominant_cat = max(category_importance.items(), key=lambda x: x[1]['pct'])
        log(f"DOMINANT CATEGORY: {dominant_cat[0]} ({dominant_cat[1]['pct']:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Logistic Regression Baseline (Coefficients)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: LOGISTIC REGRESSION BASELINE", f)
        log("-" * 40, f)
        log("", f)

        log("Fitting Logistic Regression for interpretable coefficients...", f)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logistic_model = LogisticRegression(penalty=None, max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)

        logistic_pred = logistic_model.predict_proba(X_test_scaled)[:, 1]
        logistic_metrics = evaluate_predictions(y_test, logistic_pred, "Logistic")

        log("", f)
        log("LOGISTIC REGRESSION PERFORMANCE:", f)
        log(f"  Log Loss: {logistic_metrics['log_loss']:.6f}", f)
        log(f"  AUC:      {logistic_metrics['auc']:.6f}", f)
        log(f"  Brier:    {logistic_metrics['brier']:.6f}", f)
        log("", f)

        log("COEFFICIENTS (standardized features):", f)
        log(f"  Intercept: {logistic_model.intercept_[0]:.6f}", f)
        log("", f)
        log(f"  {'Feature':<25} {'Coefficient':<12} {'Direction':<15}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*15}", f)

        coef_dict = dict(zip(feature_cols, logistic_model.coef_[0]))
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in sorted_coefs:
            direction = "+" if coef > 0 else "-"
            effect = "increases CTR" if coef > 0 else "decreases CTR"
            log(f"  {feat:<25} {coef:>+12.6f} {effect:<15}", f)
        log("", f)

        log("COEFFICIENT INTERPRETATION:", f)
        log("  (Coefficients are for standardized features: 1 unit = 1 std dev)", f)
        log("", f)

        # Key interpretations
        pos_coef = coef_dict.get('log_position', 0)
        log(f"  log_position ({pos_coef:+.4f}):", f)
        if pos_coef < 0:
            log(f"    Higher position (worse rank) DECREASES click probability.", f)
            log(f"    Interpretation: Position decay effect is real.", f)
        else:
            log(f"    Position has weak/positive effect (counterintuitive).", f)
        log("", f)

        quality_coef = coef_dict.get('quality_filled', 0)
        log(f"  quality_filled ({quality_coef:+.4f}):", f)
        if quality_coef > 0:
            log(f"    Higher quality INCREASES click probability.", f)
        else:
            log(f"    Quality has negative/weak effect.", f)
        log("", f)

        bid_coef = coef_dict.get('log_bid', 0)
        log(f"  log_bid ({bid_coef:+.4f}):", f)
        if bid_coef > 0:
            log(f"    Higher bid INCREASES click probability.", f)
        else:
            log(f"    Bid has negative/weak effect.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Model Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: MODEL COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        log(f"  {'Model':<25} {'Log Loss':<12} {'AUC':<12} {'Brier':<12}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}", f)
        log(f"  {model_type:<25} {test_metrics['log_loss']:<12.6f} {test_metrics['auc']:<12.6f} {test_metrics['brier']:<12.6f}", f)
        log(f"  {'Logistic Regression':<25} {logistic_metrics['log_loss']:<12.6f} {logistic_metrics['auc']:<12.6f} {logistic_metrics['brier']:<12.6f}", f)
        log("", f)

        if test_metrics['auc'] > logistic_metrics['auc']:
            auc_diff = test_metrics['auc'] - logistic_metrics['auc']
            log(f"{model_type} outperforms Logistic Regression by {auc_diff:.4f} AUC.", f)
        else:
            log("Logistic Regression performs comparably or better.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 10: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("DATA:", f)
        log(f"  - {len(items_df):,} item impressions", f)
        log(f"  - {items_df['clicked'].mean()*100:.2f}% CTR", f)
        log(f"  - {len(feature_cols)} features used", f)
        log("", f)

        log("MODEL PERFORMANCE:", f)
        log(f"  - {model_type} Test AUC: {test_metrics['auc']:.4f}", f)
        log(f"  - {model_type} Test Log Loss: {test_metrics['log_loss']:.4f}", f)
        log(f"  - Logistic Test AUC: {logistic_metrics['auc']:.4f}", f)
        log("", f)

        log("TOP 5 FEATURES BY IMPORTANCE:", f)
        for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
            pct = imp / total_importance * 100
            log(f"  {i}. {feat}: {pct:.1f}%", f)
        log("", f)

        log("FEATURE CATEGORY BREAKDOWN:", f)
        for cat, vals in sorted(category_importance.items(), key=lambda x: x[1]['pct'], reverse=True):
            log(f"  - {cat}: {vals['pct']:.1f}%", f)
        log("", f)

        log("KEY FINDINGS:", f)

        # Position vs Quality vs Bid dominance
        pos_pct = category_importance.get('position', {}).get('pct', 0)
        qual_pct = category_importance.get('product_quality', {}).get('pct', 0)
        bid_pct = category_importance.get('product_bid', {}).get('pct', 0)

        if pos_pct > qual_pct and pos_pct > bid_pct:
            log(f"  - POSITION features dominate ({pos_pct:.1f}% of importance)", f)
            log("    Position is the primary driver of CTR.", f)
        elif qual_pct > pos_pct and qual_pct > bid_pct:
            log(f"  - QUALITY features dominate ({qual_pct:.1f}% of importance)", f)
            log("    Product quality is the primary driver of CTR.", f)
        elif bid_pct > pos_pct and bid_pct > qual_pct:
            log(f"  - BID features dominate ({bid_pct:.1f}% of importance)", f)
            log("    Bid amount is the primary driver of CTR.", f)
        else:
            log("  - Multiple feature categories contribute significantly.", f)
        log("", f)

        # Coefficient direction check
        log("COEFFICIENT DIRECTIONS:", f)
        if coef_dict.get('log_position', 0) < 0:
            log("  - log_position: NEGATIVE (position decay confirmed)", f)
        else:
            log("  - log_position: POSITIVE/ZERO (no position decay)", f)

        if coef_dict.get('quality_filled', 0) > 0:
            log("  - quality: POSITIVE (quality increases CTR)", f)
        else:
            log("  - quality: NEGATIVE/ZERO", f)

        if coef_dict.get('log_bid', 0) > 0:
            log("  - bid: POSITIVE (bid increases CTR)", f)
        else:
            log("  - bid: NEGATIVE/ZERO", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
