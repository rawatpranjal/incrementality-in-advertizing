#!/usr/bin/env python3
"""
Click Model Comparison

Compares three click models:
1. Position-Based Model (PBM): P(C_k) = θ_k × α_i
2. Dynamic Bayesian Network (DBN): Allows multi-click via satisfaction parameter
3. Simplified DBN (SDBN): Click terminates session (cascade-like)

Comparison Metrics:
- Log-likelihood / Perplexity
- AUC-ROC
- Brier Score
- Position effect estimates (θ_k)
- Out-of-sample prediction RMSE

This analysis helps determine which model best fits the observed click behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "08_model_comparison.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# IMPORT MODELS (inline simplified versions)
# =============================================================================

def estimate_pbm(items_df, max_position=20, min_impressions=5):
    """Estimate PBM via EM (simplified version)."""
    df = items_df[items_df['position'] <= max_position].copy()

    # Get valid items
    item_counts = df.groupby('product_id').size()
    valid_items = item_counts[item_counts >= min_impressions].index
    df = df[df['product_id'].isin(valid_items)]

    item_to_idx = {item: idx for idx, item in enumerate(valid_items)}
    n_items = len(item_to_idx)
    n_positions = max_position

    # Initialize
    theta = np.ones(n_positions)
    for k in range(1, n_positions):
        theta[k] = max(0.1, 1.0 - 0.03 * k)

    alpha = np.zeros(n_items)
    for item, idx in item_to_idx.items():
        item_data = df[df['product_id'] == item]
        alpha[idx] = item_data['clicked'].mean() + 0.01

    alpha = np.clip(alpha, 0.001, 0.999)

    df['item_idx'] = df['product_id'].map(item_to_idx)
    df = df.dropna(subset=['item_idx'])
    df['item_idx'] = df['item_idx'].astype(int)

    positions = df['position'].values - 1
    item_indices = df['item_idx'].values
    clicks = df['clicked'].values.astype(float)

    # EM iterations
    for _ in range(50):
        new_theta = np.zeros(n_positions)
        for k in range(n_positions):
            mask = positions == k
            if mask.sum() > 0:
                num = clicks[mask].sum()
                denom = alpha[item_indices[mask]].sum()
                if denom > 0:
                    new_theta[k] = num / denom

        if new_theta[0] > 0:
            new_theta = new_theta / new_theta[0]
        new_theta = np.clip(new_theta, 0.001, 1.0)

        new_alpha = np.zeros(n_items)
        for i in range(n_items):
            mask = item_indices == i
            if mask.sum() > 0:
                num = clicks[mask].sum()
                denom = theta[positions[mask]].sum()
                if denom > 0:
                    new_alpha[i] = num / denom

        new_alpha = np.clip(new_alpha, 0.001, 0.999)

        if np.max(np.abs(new_theta - theta)) < 1e-6:
            break

        theta = new_theta
        alpha = new_alpha

    return theta, alpha, item_to_idx


def estimate_dbn(sessions_df, max_position=20, min_impressions=5):
    """Estimate DBN via EM (simplified version)."""
    all_items = set()
    for items in sessions_df['items']:
        all_items.update(items[:max_position])

    item_counts = {}
    for _, row in sessions_df.iterrows():
        for item in row['items'][:max_position]:
            item_counts[item] = item_counts.get(item, 0) + 1

    valid_items = {item for item, count in item_counts.items() if count >= min_impressions}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
    n_items = len(item_to_idx)

    alpha = np.full(n_items, 0.03)
    sigma = 0.5
    gamma = 0.8

    sessions_data = []
    for _, row in sessions_df.iterrows():
        items = row['items'][:max_position]
        clicks = row['clicks'][:max_position]
        item_indices = []
        valid_clicks = []
        for item, click in zip(items, clicks):
            if item in item_to_idx:
                item_indices.append(item_to_idx[item])
                valid_clicks.append(click)
        if len(item_indices) > 0:
            sessions_data.append({
                'item_indices': np.array(item_indices),
                'clicks': np.array(valid_clicks),
                'n_positions': len(item_indices)
            })

    for iteration in range(30):
        alpha_num = np.zeros(n_items)
        alpha_denom = np.zeros(n_items)
        sigma_num = 0.0
        sigma_denom = 0.0

        for session in sessions_data:
            item_idx = session['item_indices']
            clicks = session['clicks']
            n_pos = session['n_positions']

            exam_prob = np.zeros(n_pos)
            exam_prob[0] = 1.0

            for k in range(1, n_pos):
                prev_click = clicks[k-1]
                if prev_click == 0:
                    exam_prob[k] = exam_prob[k-1]
                else:
                    exam_prob[k] = exam_prob[k-1] * (1 - sigma) * gamma

            for k in range(n_pos):
                alpha_num[item_idx[k]] += clicks[k]
                alpha_denom[item_idx[k]] += exam_prob[k]
                if clicks[k] == 1:
                    sigma_denom += 1
                    if k == n_pos - 1 or clicks[k+1:].sum() == 0:
                        sigma_num += sigma

        new_alpha = np.where(alpha_denom > 0, alpha_num / alpha_denom, alpha)
        new_alpha = np.clip(new_alpha, 0.001, 0.999)
        new_sigma = sigma_num / sigma_denom if sigma_denom > 0 else sigma
        new_sigma = np.clip(new_sigma, 0.01, 0.99)

        if np.max(np.abs(new_alpha - alpha)) < 1e-6:
            break

        alpha = new_alpha
        sigma = new_sigma

    # Compute theta
    avg_alpha = np.mean(alpha)
    theta = np.ones(max_position)
    for k in range(1, max_position):
        theta[k] = theta[k-1] * ((1 - avg_alpha) + avg_alpha * (1 - sigma) * gamma)

    return theta, alpha, sigma, gamma, item_to_idx


def estimate_sdbn(sessions_df, max_position=20, min_impressions=5):
    """Estimate SDBN via MLE."""
    all_items = set()
    for items in sessions_df['items']:
        all_items.update(items[:max_position])

    item_counts = {}
    for _, row in sessions_df.iterrows():
        for item in row['items'][:max_position]:
            item_counts[item] = item_counts.get(item, 0) + 1

    valid_items = {item for item, count in item_counts.items() if count >= min_impressions}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
    n_items = len(item_to_idx)

    valid_sessions = sessions_df[sessions_df['n_clicks'] <= 1]
    multi_click = sessions_df[sessions_df['n_clicks'] > 1]

    item_impressions = np.zeros(n_items)
    item_clicks = np.zeros(n_items)

    for _, row in valid_sessions.iterrows():
        items = row['items'][:max_position]
        clicks = row['clicks'][:max_position]
        for k, (item, click) in enumerate(zip(items, clicks)):
            if item not in item_to_idx:
                continue
            idx = item_to_idx[item]
            if click == 1:
                item_impressions[idx] += 1
                item_clicks[idx] += 1
                break
            else:
                item_impressions[idx] += 1

    for _, row in multi_click.iterrows():
        items = row['items'][:max_position]
        clicks = row['clicks'][:max_position]
        for k, (item, click) in enumerate(zip(items, clicks)):
            if item not in item_to_idx:
                continue
            idx = item_to_idx[item]
            if click == 1:
                item_impressions[idx] += 1
                item_clicks[idx] += 1
                break
            else:
                item_impressions[idx] += 1

    alpha = np.where(item_impressions > 0, item_clicks / item_impressions, 0.01)
    alpha = np.clip(alpha, 0.001, 0.999)

    avg_alpha = np.mean(alpha)
    theta = np.ones(max_position)
    for k in range(1, max_position):
        theta[k] = theta[k-1] * (1 - avg_alpha)

    return theta, alpha, item_to_idx


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("CLICK MODEL COMPARISON", f)
        log("=" * 80, f)
        log("", f)

        log("MODELS COMPARED:", f)
        log("  1. PBM (Position-Based Model): P(C_k) = θ_k × α_i", f)
        log("  2. DBN (Dynamic Bayesian Network): Multi-click via satisfaction", f)
        log("  3. SDBN (Simplified DBN): Click terminates session", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Load Data
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        items_df = pd.read_parquet(DATA_DIR / "session_items.parquet")
        sessions_df = pd.read_parquet(DATA_DIR / "sessions.parquet")

        log(f"Total impressions: {len(items_df):,}", f)
        log(f"Total sessions: {len(sessions_df):,}", f)
        log(f"Total clicks: {items_df['clicked'].sum():,}", f)
        log(f"Overall CTR: {items_df['clicked'].mean()*100:.3f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Train/Test Split
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: TRAIN/TEST SPLIT", f)
        log("-" * 40, f)
        log("", f)

        # Split sessions
        train_sessions, test_sessions = train_test_split(
            sessions_df, test_size=0.2, random_state=42
        )

        # Get corresponding items
        train_auction_ids = set(train_sessions['auction_id'])
        test_auction_ids = set(test_sessions['auction_id'])

        train_items = items_df[items_df['auction_id'].isin(train_auction_ids)]
        test_items = items_df[items_df['auction_id'].isin(test_auction_ids)]

        log(f"Train sessions: {len(train_sessions):,}", f)
        log(f"Test sessions: {len(test_sessions):,}", f)
        log(f"Train items: {len(train_items):,}", f)
        log(f"Test items: {len(test_items):,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Fit Models on Training Data
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: MODEL ESTIMATION", f)
        log("-" * 40, f)
        log("", f)

        MAX_POS = 20
        MIN_IMP = 5

        # PBM
        log("Fitting PBM...", f)
        theta_pbm, alpha_pbm, item_to_idx_pbm = estimate_pbm(
            train_items, max_position=MAX_POS, min_impressions=MIN_IMP
        )
        log(f"  Items: {len(alpha_pbm):,}", f)
        log(f"  Mean α: {np.mean(alpha_pbm):.4f}", f)
        log("", f)

        # DBN
        log("Fitting DBN...", f)
        theta_dbn, alpha_dbn, sigma_dbn, gamma_dbn, item_to_idx_dbn = estimate_dbn(
            train_sessions, max_position=MAX_POS, min_impressions=MIN_IMP
        )
        log(f"  Items: {len(alpha_dbn):,}", f)
        log(f"  σ (satisfaction): {sigma_dbn:.4f}", f)
        log(f"  γ (continuation): {gamma_dbn:.4f}", f)
        log(f"  Mean α: {np.mean(alpha_dbn):.4f}", f)
        log("", f)

        # SDBN
        log("Fitting SDBN...", f)
        theta_sdbn, alpha_sdbn, item_to_idx_sdbn = estimate_sdbn(
            train_sessions, max_position=MAX_POS, min_impressions=MIN_IMP
        )
        log(f"  Items: {len(alpha_sdbn):,}", f)
        log(f"  Mean α: {np.mean(alpha_sdbn):.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Position Effects Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: POSITION EFFECTS (θ_k) COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        # Also compute empirical baseline
        ctr_by_pos = train_items[train_items['position'] <= MAX_POS].groupby('position')['clicked'].mean()
        theta_empirical = ctr_by_pos / ctr_by_pos.iloc[0] if len(ctr_by_pos) > 0 else pd.Series()

        log(f"  {'Position':<10} {'PBM':<10} {'DBN':<10} {'SDBN':<10} {'Empirical':<10}", f)
        log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", f)

        for k in range(min(MAX_POS, 15)):
            emp_val = theta_empirical.iloc[k] if k < len(theta_empirical) else np.nan
            log(f"  {k+1:<10} {theta_pbm[k]:<10.3f} {theta_dbn[k]:<10.3f} {theta_sdbn[k]:<10.3f} {emp_val:<10.3f}", f)
        log("", f)

        # Summary statistics
        log("Position effect summary:", f)
        log(f"  θ_5/θ_1:  PBM={theta_pbm[4]:.3f}, DBN={theta_dbn[4]:.3f}, SDBN={theta_sdbn[4]:.3f}", f)
        log(f"  θ_10/θ_1: PBM={theta_pbm[9]:.3f}, DBN={theta_dbn[9]:.3f}, SDBN={theta_sdbn[9]:.3f}", f)
        log("", f)

        # Correlation between model theta estimates
        log("Correlation between θ estimates:", f)
        corr_pbm_dbn = np.corrcoef(theta_pbm, theta_dbn)[0,1]
        corr_pbm_sdbn = np.corrcoef(theta_pbm, theta_sdbn)[0,1]
        corr_dbn_sdbn = np.corrcoef(theta_dbn, theta_sdbn)[0,1]
        log(f"  PBM vs DBN: {corr_pbm_dbn:.3f}", f)
        log(f"  PBM vs SDBN: {corr_pbm_sdbn:.3f}", f)
        log(f"  DBN vs SDBN: {corr_dbn_sdbn:.3f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Prediction Functions
        # -----------------------------------------------------------------
        def predict_pbm(row, theta, alpha, item_to_idx):
            if row['position'] > len(theta):
                return np.mean(alpha)
            theta_k = theta[int(row['position']) - 1]
            if row['product_id'] in item_to_idx:
                alpha_i = alpha[item_to_idx[row['product_id']]]
            else:
                alpha_i = np.mean(alpha)
            return theta_k * alpha_i

        def predict_dbn(row, theta, alpha, item_to_idx):
            if row['position'] > len(theta):
                return np.mean(alpha)
            theta_k = theta[int(row['position']) - 1]
            if row['product_id'] in item_to_idx:
                alpha_i = alpha[item_to_idx[row['product_id']]]
            else:
                alpha_i = np.mean(alpha)
            return theta_k * alpha_i

        def predict_sdbn(row, theta, alpha, item_to_idx):
            if row['position'] > len(theta):
                return np.mean(alpha)
            theta_k = theta[int(row['position']) - 1]
            if row['product_id'] in item_to_idx:
                alpha_i = alpha[item_to_idx[row['product_id']]]
            else:
                alpha_i = np.mean(alpha)
            return theta_k * alpha_i

        # -----------------------------------------------------------------
        # Section 6: Out-of-Sample Evaluation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: OUT-OF-SAMPLE EVALUATION", f)
        log("-" * 40, f)
        log("", f)

        test_items_eval = test_items[test_items['position'] <= MAX_POS].copy()

        log("Generating predictions...", f)

        # PBM predictions
        test_items_eval['pred_pbm'] = test_items_eval.apply(
            lambda row: predict_pbm(row, theta_pbm, alpha_pbm, item_to_idx_pbm), axis=1
        )

        # DBN predictions
        test_items_eval['pred_dbn'] = test_items_eval.apply(
            lambda row: predict_dbn(row, theta_dbn, alpha_dbn, item_to_idx_dbn), axis=1
        )

        # SDBN predictions
        test_items_eval['pred_sdbn'] = test_items_eval.apply(
            lambda row: predict_sdbn(row, theta_sdbn, alpha_sdbn, item_to_idx_sdbn), axis=1
        )

        # Empirical baseline (position-only)
        test_items_eval['pred_baseline'] = test_items_eval['position'].map(
            lambda p: ctr_by_pos.iloc[int(p)-1] if int(p) <= len(ctr_by_pos) else ctr_by_pos.mean()
        )

        # Clip predictions
        for col in ['pred_pbm', 'pred_dbn', 'pred_sdbn', 'pred_baseline']:
            test_items_eval[col] = test_items_eval[col].clip(1e-6, 1-1e-6)

        y_true = test_items_eval['clicked'].values

        log("", f)
        log("PREDICTION METRICS (Out-of-Sample):", f)
        log(f"  {'Metric':<20} {'Baseline':<12} {'PBM':<12} {'DBN':<12} {'SDBN':<12}", f)
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

        # Log Loss
        ll_baseline = log_loss(y_true, test_items_eval['pred_baseline'])
        ll_pbm = log_loss(y_true, test_items_eval['pred_pbm'])
        ll_dbn = log_loss(y_true, test_items_eval['pred_dbn'])
        ll_sdbn = log_loss(y_true, test_items_eval['pred_sdbn'])
        log(f"  {'Log Loss':<20} {ll_baseline:<12.4f} {ll_pbm:<12.4f} {ll_dbn:<12.4f} {ll_sdbn:<12.4f}", f)

        # Brier Score
        bs_baseline = brier_score_loss(y_true, test_items_eval['pred_baseline'])
        bs_pbm = brier_score_loss(y_true, test_items_eval['pred_pbm'])
        bs_dbn = brier_score_loss(y_true, test_items_eval['pred_dbn'])
        bs_sdbn = brier_score_loss(y_true, test_items_eval['pred_sdbn'])
        log(f"  {'Brier Score':<20} {bs_baseline:<12.4f} {bs_pbm:<12.4f} {bs_dbn:<12.4f} {bs_sdbn:<12.4f}", f)

        # AUC
        auc_baseline = roc_auc_score(y_true, test_items_eval['pred_baseline'])
        auc_pbm = roc_auc_score(y_true, test_items_eval['pred_pbm'])
        auc_dbn = roc_auc_score(y_true, test_items_eval['pred_dbn'])
        auc_sdbn = roc_auc_score(y_true, test_items_eval['pred_sdbn'])
        log(f"  {'AUC':<20} {auc_baseline:<12.4f} {auc_pbm:<12.4f} {auc_dbn:<12.4f} {auc_sdbn:<12.4f}", f)

        # RMSE
        rmse_baseline = np.sqrt(mean_squared_error(y_true, test_items_eval['pred_baseline']))
        rmse_pbm = np.sqrt(mean_squared_error(y_true, test_items_eval['pred_pbm']))
        rmse_dbn = np.sqrt(mean_squared_error(y_true, test_items_eval['pred_dbn']))
        rmse_sdbn = np.sqrt(mean_squared_error(y_true, test_items_eval['pred_sdbn']))
        log(f"  {'RMSE':<20} {rmse_baseline:<12.4f} {rmse_pbm:<12.4f} {rmse_dbn:<12.4f} {rmse_sdbn:<12.4f}", f)

        log("", f)

        # Relative improvement
        log("Relative improvement over baseline:", f)
        log(f"  PBM:  Log Loss {(ll_baseline-ll_pbm)/ll_baseline*100:.2f}%, AUC {(auc_pbm-auc_baseline)/auc_baseline*100:.2f}%", f)
        log(f"  DBN:  Log Loss {(ll_baseline-ll_dbn)/ll_baseline*100:.2f}%, AUC {(auc_dbn-auc_baseline)/auc_baseline*100:.2f}%", f)
        log(f"  SDBN: Log Loss {(ll_baseline-ll_sdbn)/ll_baseline*100:.2f}%, AUC {(auc_sdbn-auc_baseline)/auc_baseline*100:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Perplexity Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: PERPLEXITY ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("Perplexity = exp(average log loss per prediction)", f)
        log("Lower perplexity indicates better model fit", f)
        log("", f)

        perp_baseline = np.exp(ll_baseline)
        perp_pbm = np.exp(ll_pbm)
        perp_dbn = np.exp(ll_dbn)
        perp_sdbn = np.exp(ll_sdbn)

        log(f"  Baseline: {perp_baseline:.4f}", f)
        log(f"  PBM:      {perp_pbm:.4f}", f)
        log(f"  DBN:      {perp_dbn:.4f}", f)
        log(f"  SDBN:     {perp_sdbn:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Stratified Evaluation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: STRATIFIED EVALUATION BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        for placement in sorted(test_items_eval['placement'].unique()):
            subset = test_items_eval[test_items_eval['placement'] == placement]
            if len(subset) < 500:
                continue

            y_sub = subset['clicked'].values

            log(f"PLACEMENT {placement} (N={len(subset):,}):", f)

            try:
                auc_pbm_sub = roc_auc_score(y_sub, subset['pred_pbm'])
                auc_dbn_sub = roc_auc_score(y_sub, subset['pred_dbn'])
                auc_sdbn_sub = roc_auc_score(y_sub, subset['pred_sdbn'])

                log(f"  AUC: PBM={auc_pbm_sub:.4f}, DBN={auc_dbn_sub:.4f}, SDBN={auc_sdbn_sub:.4f}", f)

                ll_pbm_sub = log_loss(y_sub, subset['pred_pbm'])
                ll_dbn_sub = log_loss(y_sub, subset['pred_dbn'])
                ll_sdbn_sub = log_loss(y_sub, subset['pred_sdbn'])

                log(f"  Log Loss: PBM={ll_pbm_sub:.4f}, DBN={ll_dbn_sub:.4f}, SDBN={ll_sdbn_sub:.4f}", f)
            except:
                log(f"  Could not compute metrics", f)

            log("", f)

        # -----------------------------------------------------------------
        # Section 9: Item Effect Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: ITEM EFFECT (α) COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        log("Correlation between α estimates:", f)

        # Find common items
        common_items = set(item_to_idx_pbm.keys()) & set(item_to_idx_dbn.keys()) & set(item_to_idx_sdbn.keys())
        log(f"  Common items: {len(common_items):,}", f)

        if len(common_items) > 100:
            alpha_pbm_common = np.array([alpha_pbm[item_to_idx_pbm[item]] for item in common_items])
            alpha_dbn_common = np.array([alpha_dbn[item_to_idx_dbn[item]] for item in common_items])
            alpha_sdbn_common = np.array([alpha_sdbn[item_to_idx_sdbn[item]] for item in common_items])

            corr_alpha_pbm_dbn = np.corrcoef(alpha_pbm_common, alpha_dbn_common)[0,1]
            corr_alpha_pbm_sdbn = np.corrcoef(alpha_pbm_common, alpha_sdbn_common)[0,1]
            corr_alpha_dbn_sdbn = np.corrcoef(alpha_dbn_common, alpha_sdbn_common)[0,1]

            log(f"  PBM vs DBN: {corr_alpha_pbm_dbn:.3f}", f)
            log(f"  PBM vs SDBN: {corr_alpha_pbm_sdbn:.3f}", f)
            log(f"  DBN vs SDBN: {corr_alpha_dbn_sdbn:.3f}", f)
        else:
            log("  Insufficient common items for comparison", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 10: Summary Table
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("MODEL COMPARISON SUMMARY", f)
        log("=" * 60, f)
        log("", f)

        log(f"{'Model':<15} {'Log Loss':<12} {'AUC':<10} {'Perplexity':<12} {'Best?':<8}", f)
        log(f"{'-'*15} {'-'*12} {'-'*10} {'-'*12} {'-'*8}", f)

        best_ll = min(ll_pbm, ll_dbn, ll_sdbn)
        best_auc = max(auc_pbm, auc_dbn, auc_sdbn)

        def best_marker(val, best, higher_better=False):
            if higher_better:
                return '***' if val == best else ''
            else:
                return '***' if val == best else ''

        log(f"{'Baseline':<15} {ll_baseline:<12.4f} {auc_baseline:<10.4f} {perp_baseline:<12.4f} {'':<8}", f)
        log(f"{'PBM':<15} {ll_pbm:<12.4f} {auc_pbm:<10.4f} {perp_pbm:<12.4f} {best_marker(ll_pbm, best_ll):<8}", f)
        log(f"{'DBN':<15} {ll_dbn:<12.4f} {auc_dbn:<10.4f} {perp_dbn:<12.4f} {best_marker(ll_dbn, best_ll):<8}", f)
        log(f"{'SDBN':<15} {ll_sdbn:<12.4f} {auc_sdbn:<10.4f} {perp_sdbn:<12.4f} {best_marker(ll_sdbn, best_ll):<8}", f)

        log("", f)
        log("*** = Best model for that metric", f)
        log("", f)

        log("POSITION EFFECT SUMMARY:", f)
        log(f"  Position 5 examination (θ_5):", f)
        log(f"    PBM:  {theta_pbm[4]:.3f} ({theta_pbm[4]*100:.1f}% of pos 1)", f)
        log(f"    DBN:  {theta_dbn[4]:.3f} ({theta_dbn[4]*100:.1f}% of pos 1)", f)
        log(f"    SDBN: {theta_sdbn[4]:.3f} ({theta_sdbn[4]*100:.1f}% of pos 1)", f)
        log("", f)
        log(f"  Position 10 examination (θ_10):", f)
        log(f"    PBM:  {theta_pbm[9]:.3f} ({theta_pbm[9]*100:.1f}% of pos 1)", f)
        log(f"    DBN:  {theta_dbn[9]:.3f} ({theta_dbn[9]*100:.1f}% of pos 1)", f)
        log(f"    SDBN: {theta_sdbn[9]:.3f} ({theta_sdbn[9]*100:.1f}% of pos 1)", f)
        log("", f)

        log("KEY FINDINGS:", f)

        # Determine best model
        if ll_dbn < ll_pbm and ll_dbn < ll_sdbn:
            best_model = "DBN"
        elif ll_pbm < ll_sdbn:
            best_model = "PBM"
        else:
            best_model = "SDBN"

        log(f"  1. Best model by log loss: {best_model}", f)

        # Multi-click handling
        multi_click_rate = (sessions_df['n_clicks'] >= 2).mean()
        log(f"  2. Multi-click rate: {multi_click_rate*100:.1f}%", f)
        if multi_click_rate > 0.05:
            log("     -> DBN may be more appropriate (handles multi-click)", f)
        else:
            log("     -> SDBN assumption (single click) approximately holds", f)

        # Position effect monotonicity
        mono_pbm = sum(1 for i in range(len(theta_pbm)-1) if theta_pbm[i] >= theta_pbm[i+1]) / (len(theta_pbm)-1)
        mono_dbn = sum(1 for i in range(len(theta_dbn)-1) if theta_dbn[i] >= theta_dbn[i+1]) / (len(theta_dbn)-1)
        mono_sdbn = sum(1 for i in range(len(theta_sdbn)-1) if theta_sdbn[i] >= theta_sdbn[i+1]) / (len(theta_sdbn)-1)

        log(f"  3. Position effect monotonicity:", f)
        log(f"     PBM:  {mono_pbm*100:.1f}%", f)
        log(f"     DBN:  {mono_dbn*100:.1f}%", f)
        log(f"     SDBN: {mono_sdbn*100:.1f}%", f)

        log("", f)

        log("RECOMMENDATIONS:", f)
        log(f"  - For CTR prediction: Use {best_model} (best log loss)", f)
        log(f"  - For position effect estimation: SDBN provides clearest decay curve", f)
        log(f"  - For handling multi-click: DBN with σ={sigma_dbn:.3f}, γ={gamma_dbn:.3f}", f)
        log("", f)

        log("=" * 80, f)
        log("MODEL COMPARISON COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
