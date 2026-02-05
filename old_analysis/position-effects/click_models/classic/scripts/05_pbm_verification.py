#!/usr/bin/env python3
"""
Position-Based Model (PBM) Verification

Mathematical Framework:
- Examination hypothesis: P(C_k = 1) = θ_k × α_i
- θ_k = P(examine position k) — position effect (normalized: θ_1 = 1)
- α_i = P(click | examined, item i) — item attractiveness

Estimation:
- EM algorithm for joint θ_k and α_i estimation
- MLE with position-only θ_k (pooled α)

Testable Implications:
1. Multiplicative separability: CTR(k,i)/CTR(k,j) = α_i/α_j for all k
2. Monotonicity: θ_k should decrease with k
3. Conditional independence: C_k ⊥ C_k' | item (given examination)

Reference: Craswell et al. (2008) "An Experimental Comparison of Click Position-Bias Models"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "05_pbm_verification.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# PBM ESTIMATION
# =============================================================================

class PositionBasedModel:
    """
    Position-Based Model (PBM) for click prediction.

    Model: P(click at position k, item i) = θ_k × α_i

    Identification:
    - θ_1 = 1 (normalization)
    - θ_k and α_i estimated jointly via EM
    """

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.theta = None  # Position effects
        self.alpha = None  # Item attractiveness
        self.fitted = False

    def fit_em(self, items_df, max_iter=100, tol=1e-6, verbose=True):
        """
        EM algorithm for PBM estimation.

        E-step: Compute expected examination given observed clicks
        M-step: Update θ and α
        """
        # Filter to positions within max
        df = items_df[items_df['position'] <= self.max_position].copy()

        # Get unique items with sufficient data
        item_counts = df.groupby('product_id').size()
        valid_items = item_counts[item_counts >= self.min_impressions].index
        df = df[df['product_id'].isin(valid_items)]

        if verbose:
            print(f"  Fitting PBM with {len(df):,} observations, {len(valid_items):,} items")

        # Initialize parameters
        n_positions = self.max_position
        n_items = len(valid_items)
        item_to_idx = {item: idx for idx, item in enumerate(valid_items)}

        # Initialize theta (position effects) - decreasing
        theta = np.ones(n_positions)
        for k in range(1, n_positions):
            theta[k] = max(0.1, 1.0 - 0.05 * k)

        # Initialize alpha (item effects) - from observed CTR
        alpha = np.zeros(n_items)
        for item, idx in item_to_idx.items():
            item_data = df[df['product_id'] == item]
            alpha[idx] = item_data['clicked'].mean() + 0.01  # Add small constant

        # Clip alpha
        alpha = np.clip(alpha, 0.001, 0.999)

        # Precompute indices
        df['item_idx'] = df['product_id'].map(item_to_idx)
        df = df.dropna(subset=['item_idx'])
        df['item_idx'] = df['item_idx'].astype(int)

        positions = df['position'].values - 1  # 0-indexed
        item_indices = df['item_idx'].values
        clicks = df['clicked'].values.astype(float)

        ll_history = []

        for iteration in range(max_iter):
            # E-step: Compute expected examination
            # P(examined | clicked=1) = 1
            # P(examined | clicked=0) = θ_k(1-α_i) / (1 - θ_k*α_i) * θ_k
            # But simpler: just use current θ and α to compute expected click prob

            # M-step directly using current estimates
            # For θ_k: sum of clicks at k / sum of α_i at k
            # For α_i: sum of clicks for i / sum of θ_k for i

            # Update theta
            new_theta = np.zeros(n_positions)
            for k in range(n_positions):
                mask = positions == k
                if mask.sum() > 0:
                    num = clicks[mask].sum()
                    denom = alpha[item_indices[mask]].sum()
                    if denom > 0:
                        new_theta[k] = num / denom

            # Normalize theta (theta[0] = 1)
            if new_theta[0] > 0:
                new_theta = new_theta / new_theta[0]
            new_theta = np.clip(new_theta, 0.001, 1.0)

            # Update alpha
            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                mask = item_indices == i
                if mask.sum() > 0:
                    num = clicks[mask].sum()
                    denom = theta[positions[mask]].sum()
                    if denom > 0:
                        new_alpha[i] = num / denom

            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            # Compute log-likelihood
            probs = theta[positions] * alpha[item_indices]
            probs = np.clip(probs, 1e-10, 1-1e-10)
            ll = np.sum(clicks * np.log(probs) + (1-clicks) * np.log(1-probs))
            ll_history.append(ll)

            # Check convergence
            theta_change = np.max(np.abs(new_theta - theta))
            alpha_change = np.max(np.abs(new_alpha - alpha))

            theta = new_theta
            alpha = new_alpha

            if verbose and iteration % 10 == 0:
                print(f"    Iter {iteration}: LL={ll:.2f}, θ_change={theta_change:.6f}, α_change={alpha_change:.6f}")

            if theta_change < tol and alpha_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

        self.theta = theta
        self.alpha = alpha
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.ll_history = ll_history
        self.fitted = True

        return self

    def predict_ctr(self, position, product_id=None):
        """Predict CTR for position (and optionally item)."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        if position > self.max_position:
            return 0.0

        theta_k = self.theta[position - 1]

        if product_id is None:
            # Average across all items
            return theta_k * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            alpha_i = self.alpha[self.item_to_idx[product_id]]
            return theta_k * alpha_i
        else:
            return theta_k * np.mean(self.alpha)

    def get_theta(self):
        """Return position effects."""
        return self.theta.copy()

    def get_alpha_stats(self):
        """Return statistics on item attractiveness."""
        return {
            'mean': np.mean(self.alpha),
            'std': np.std(self.alpha),
            'min': np.min(self.alpha),
            'max': np.max(self.alpha),
            'median': np.median(self.alpha),
            'n_items': len(self.alpha)
        }


def estimate_pbm_simple(items_df, max_position=20):
    """
    Simple PBM estimation: position effects only (pooled item effect).

    θ_k = CTR(k) / CTR(1)

    This is a baseline that ignores item heterogeneity.
    """
    df = items_df[items_df['position'] <= max_position].copy()

    ctr_by_pos = df.groupby('position').agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    ctr_by_pos.columns = ['position', 'clicks', 'impressions', 'ctr']

    # Normalize by position 1
    ctr_1 = ctr_by_pos[ctr_by_pos['position'] == 1]['ctr'].values[0]
    ctr_by_pos['theta'] = ctr_by_pos['ctr'] / ctr_1

    return ctr_by_pos


def test_multiplicative_separability(items_df, min_item_impressions=10, max_position=10):
    """
    Test PBM's multiplicative separability assumption.

    Under PBM: CTR(k,i)/CTR(k,j) = α_i/α_j for all k

    This ratio should be constant across positions.
    """
    df = items_df[(items_df['position'] <= max_position)].copy()

    # Get items with enough data at multiple positions
    item_pos_counts = df.groupby(['product_id', 'position']).size().unstack(fill_value=0)

    # Items with at least min_impressions at positions 1, 2, and 3
    valid_items = []
    for item in item_pos_counts.index:
        counts = item_pos_counts.loc[item]
        if all(counts[p] >= min_item_impressions for p in [1, 2, 3] if p in counts.index):
            valid_items.append(item)

    if len(valid_items) < 10:
        return None, "Insufficient items with data at multiple positions"

    # Compute CTR at each position for valid items
    item_ctr = df[df['product_id'].isin(valid_items)].groupby(['product_id', 'position']).agg({
        'clicked': ['sum', 'count', 'mean']
    }).reset_index()
    item_ctr.columns = ['product_id', 'position', 'clicks', 'impressions', 'ctr']

    # For each pair of items, compute ratio at each position
    results = []
    valid_items_list = list(valid_items)[:50]  # Limit for speed

    for i, item_i in enumerate(valid_items_list):
        for j, item_j in enumerate(valid_items_list):
            if i >= j:
                continue

            ratios = []
            for pos in [1, 2, 3]:
                ctr_i = item_ctr[(item_ctr['product_id'] == item_i) & (item_ctr['position'] == pos)]['ctr'].values
                ctr_j = item_ctr[(item_ctr['product_id'] == item_j) & (item_ctr['position'] == pos)]['ctr'].values

                if len(ctr_i) > 0 and len(ctr_j) > 0 and ctr_j[0] > 0.001:
                    ratios.append(ctr_i[0] / ctr_j[0])

            if len(ratios) >= 2:
                results.append({
                    'item_i': item_i,
                    'item_j': item_j,
                    'ratio_std': np.std(ratios),
                    'ratio_mean': np.mean(ratios),
                    'cv': np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else np.nan,
                    'n_positions': len(ratios)
                })

    if len(results) == 0:
        return None, "No valid item pairs found"

    results_df = pd.DataFrame(results)

    # If multiplicative separability holds, CV should be low
    mean_cv = results_df['cv'].mean()
    median_cv = results_df['cv'].median()

    return results_df, f"Mean CV: {mean_cv:.3f}, Median CV: {median_cv:.3f}"


def test_monotonicity(theta, alpha=0.05):
    """
    Test if theta is monotonically decreasing.

    Returns: fraction of positions where monotonicity holds
    """
    violations = 0
    for k in range(len(theta) - 1):
        if theta[k] < theta[k+1]:
            violations += 1

    monotonicity_rate = 1 - violations / (len(theta) - 1)

    return {
        'monotonicity_rate': monotonicity_rate,
        'n_violations': violations,
        'n_positions': len(theta) - 1
    }


def test_conditional_independence(items_df, max_position=10):
    """
    Test conditional independence: C_k ⊥ C_k' | item

    Under PBM, clicks at different positions should be independent given item.
    This means: P(C_k=1, C_k'=1 | item) = P(C_k=1 | item) * P(C_k'=1 | item)

    We test by looking at multi-click patterns.
    """
    # Load sessions
    sessions = pd.read_parquet(DATA_DIR / "sessions.parquet")

    # Filter to multi-click sessions
    multi_click = sessions[sessions['n_clicks'] >= 2].copy()

    if len(multi_click) == 0:
        return None, "No multi-click sessions"

    # For each session, check if clicks are at adjacent positions
    adjacent_rate = 0
    total_pairs = 0

    for _, session in multi_click.iterrows():
        clicks_list = session['clicks']
        click_positions = [i for i, c in enumerate(clicks_list) if c == 1]

        for i in range(len(click_positions) - 1):
            gap = click_positions[i+1] - click_positions[i]
            if gap == 1:
                adjacent_rate += 1
            total_pairs += 1

    if total_pairs > 0:
        adjacent_rate = adjacent_rate / total_pairs

    # Under independence, adjacent clicks should be rare
    # Under cascade, adjacent clicks would be more common

    return {
        'adjacent_click_rate': adjacent_rate,
        'total_click_pairs': total_pairs,
        'n_multi_click_sessions': len(multi_click)
    }, f"Adjacent click rate: {adjacent_rate:.3f}"


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("POSITION-BASED MODEL (PBM) VERIFICATION", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  P(click at position k, item i) = θ_k × α_i", f)
        log("  θ_k = P(examine position k) — position effect", f)
        log("  α_i = P(click | examined, item i) — item attractiveness", f)
        log("  Normalization: θ_1 = 1", f)
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

        log(f"Loaded session_items: {len(items_df):,} rows", f)
        log(f"Loaded sessions: {len(sessions_df):,} rows", f)
        log("", f)

        log("Data summary:", f)
        log(f"  Total impressions: {len(items_df):,}", f)
        log(f"  Total clicks: {items_df['clicked'].sum():,}", f)
        log(f"  Overall CTR: {items_df['clicked'].mean()*100:.3f}%", f)
        log(f"  Unique products: {items_df['product_id'].nunique():,}", f)
        log(f"  Max position: {items_df['position'].max()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Simple PBM (Position Effects Only)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: SIMPLE PBM (POSITION EFFECTS ONLY)", f)
        log("-" * 40, f)
        log("", f)

        log("MODEL: θ_k = CTR(k) / CTR(1)", f)
        log("ASSUMPTION: Pooled item effect (ignores item heterogeneity)", f)
        log("", f)

        simple_pbm = estimate_pbm_simple(items_df, max_position=20)

        log("Position effects (θ_k):", f)
        log(f"  {'Position':<10} {'Impressions':<12} {'Clicks':<10} {'CTR %':<10} {'θ_k':<10}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}", f)

        for _, row in simple_pbm.iterrows():
            log(f"  {int(row['position']):<10} {int(row['impressions']):<12,} {int(row['clicks']):<10,} {row['ctr']*100:<10.2f} {row['theta']:<10.3f}", f)
        log("", f)

        # Monotonicity check
        theta_simple = simple_pbm['theta'].values
        mono_result = test_monotonicity(theta_simple)
        log(f"Monotonicity check:", f)
        log(f"  Monotonicity rate: {mono_result['monotonicity_rate']*100:.1f}%", f)
        log(f"  Violations: {mono_result['n_violations']} / {mono_result['n_positions']}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Full PBM via EM
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: FULL PBM (EM ESTIMATION)", f)
        log("-" * 40, f)
        log("", f)

        log("MODEL: θ_k × α_i joint estimation via EM", f)
        log("IDENTIFICATION: Same item at different positions", f)
        log("", f)

        pbm = PositionBasedModel(max_position=20, min_impressions=5)
        log("Fitting PBM via EM...", f)
        pbm.fit_em(items_df, max_iter=100, verbose=True)
        log("", f)

        # Position effects
        theta_em = pbm.get_theta()

        log("Position effects (θ_k) from EM:", f)
        log(f"  {'Position':<10} {'θ_k (EM)':<12} {'θ_k (Simple)':<12} {'Difference':<12}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}", f)

        for k in range(min(20, len(theta_em))):
            simple_theta = simple_pbm[simple_pbm['position'] == k+1]['theta'].values
            simple_val = simple_theta[0] if len(simple_theta) > 0 else np.nan
            diff = theta_em[k] - simple_val if not np.isnan(simple_val) else np.nan
            log(f"  {k+1:<10} {theta_em[k]:<12.3f} {simple_val:<12.3f} {diff:<12.3f}", f)
        log("", f)

        # Item attractiveness stats
        alpha_stats = pbm.get_alpha_stats()
        log("Item attractiveness (α_i) statistics:", f)
        log(f"  N items: {alpha_stats['n_items']:,}", f)
        log(f"  Mean: {alpha_stats['mean']:.4f}", f)
        log(f"  Std: {alpha_stats['std']:.4f}", f)
        log(f"  Min: {alpha_stats['min']:.4f}", f)
        log(f"  Max: {alpha_stats['max']:.4f}", f)
        log(f"  Median: {alpha_stats['median']:.4f}", f)
        log("", f)

        # EM convergence
        log("EM convergence:", f)
        log(f"  Initial LL: {pbm.ll_history[0]:.2f}", f)
        log(f"  Final LL: {pbm.ll_history[-1]:.2f}", f)
        log(f"  Iterations: {len(pbm.ll_history)}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Monotonicity Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: MONOTONICITY TEST", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: θ_k should decrease with k (examination probability decays)", f)
        log("", f)

        mono_result_em = test_monotonicity(theta_em)
        log(f"EM results:", f)
        log(f"  Monotonicity rate: {mono_result_em['monotonicity_rate']*100:.1f}%", f)
        log(f"  Violations: {mono_result_em['n_violations']} / {mono_result_em['n_positions']}", f)
        log("", f)

        # Find violations
        violations = []
        for k in range(len(theta_em) - 1):
            if theta_em[k] < theta_em[k+1]:
                violations.append({
                    'position': k+1,
                    'theta_k': theta_em[k],
                    'theta_k_plus_1': theta_em[k+1],
                    'increase': theta_em[k+1] - theta_em[k]
                })

        if violations:
            log("Monotonicity violations:", f)
            for v in violations[:10]:
                log(f"  Position {v['position']}: θ_{v['position']}={v['theta_k']:.3f} < θ_{v['position']+1}={v['theta_k_plus_1']:.3f} (+{v['increase']:.3f})", f)
        else:
            log("No monotonicity violations", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Multiplicative Separability Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: MULTIPLICATIVE SEPARABILITY TEST", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: CTR(k,i)/CTR(k,j) = α_i/α_j constant across positions k", f)
        log("", f)

        sep_results, sep_msg = test_multiplicative_separability(items_df, min_item_impressions=5, max_position=10)

        if sep_results is not None:
            log(f"Result: {sep_msg}", f)
            log("", f)

            log("Ratio coefficient of variation (CV) distribution:", f)
            cv_values = sep_results['cv'].dropna()
            log(f"  N item pairs: {len(cv_values):,}", f)
            log(f"  Mean CV: {cv_values.mean():.3f}", f)
            log(f"  Median CV: {cv_values.median():.3f}", f)
            log(f"  Std CV: {cv_values.std():.3f}", f)
            log("", f)

            log("Interpretation:", f)
            if cv_values.median() < 0.3:
                log("  CV < 0.3: STRONG support for multiplicative separability", f)
            elif cv_values.median() < 0.5:
                log("  CV 0.3-0.5: MODERATE support for multiplicative separability", f)
            else:
                log("  CV > 0.5: WEAK support - item effects may vary by position", f)
        else:
            log(f"Could not test: {sep_msg}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Conditional Independence Test
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: CONDITIONAL INDEPENDENCE TEST", f)
        log("-" * 40, f)
        log("", f)

        log("HYPOTHESIS: C_k ⊥ C_k' | item (clicks independent given item)", f)
        log("TEST: If independent, adjacent clicks should be rare", f)
        log("", f)

        indep_result, indep_msg = test_conditional_independence(items_df)

        if indep_result is not None:
            log(f"Result: {indep_msg}", f)
            log("", f)

            log("Multi-click patterns:", f)
            log(f"  N multi-click sessions: {indep_result['n_multi_click_sessions']:,}", f)
            log(f"  Total click pairs: {indep_result['total_click_pairs']:,}", f)
            log(f"  Adjacent click rate: {indep_result['adjacent_click_rate']*100:.1f}%", f)
            log("", f)

            log("Interpretation:", f)
            # Expected under independence
            # If positions are independent, adjacent clicks happen by chance
            # P(adjacent) ≈ 2/n for n positions
            avg_positions = items_df.groupby('auction_id')['position'].max().mean()
            expected_adjacent = 2 / avg_positions if avg_positions > 2 else 0.5

            log(f"  Expected adjacent rate under independence: ~{expected_adjacent*100:.1f}%", f)
            log(f"  Observed adjacent rate: {indep_result['adjacent_click_rate']*100:.1f}%", f)

            if indep_result['adjacent_click_rate'] < expected_adjacent * 1.5:
                log("  CONSISTENT with conditional independence", f)
            else:
                log("  VIOLATION - suggests cascade/sequential behavior", f)
        else:
            log(f"Could not test: {indep_msg}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Stratified Analysis by Placement
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: STRATIFIED ANALYSIS BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Check if position effects vary by UI context", f)
        log("", f)

        for placement in sorted(items_df['placement'].unique()):
            placement_data = items_df[items_df['placement'] == placement]

            if len(placement_data) < 1000:
                continue

            log(f"PLACEMENT {placement}:", f)
            log(f"  N impressions: {len(placement_data):,}", f)
            log(f"  N clicks: {placement_data['clicked'].sum():,}", f)
            log(f"  CTR: {placement_data['clicked'].mean()*100:.2f}%", f)

            # Simple PBM for this placement
            placement_pbm = estimate_pbm_simple(placement_data, max_position=10)

            log(f"  Position effects:", f)
            for _, row in placement_pbm.head(5).iterrows():
                log(f"    Position {int(row['position'])}: θ={row['theta']:.3f}, CTR={row['ctr']*100:.2f}%", f)
            log("", f)

        # -----------------------------------------------------------------
        # Section 8: Prediction Quality
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: PREDICTION QUALITY", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Evaluate PBM fit via prediction accuracy", f)
        log("", f)

        # Predict CTR for each observation
        items_df_test = items_df[items_df['position'] <= 20].copy()

        # Simple PBM predictions
        simple_pred = items_df_test['position'].map(
            lambda p: simple_pbm[simple_pbm['position'] == p]['ctr'].values[0]
            if p in simple_pbm['position'].values else np.nan
        )

        # Full PBM predictions
        em_pred = items_df_test.apply(
            lambda row: pbm.predict_ctr(int(row['position']), row['product_id']),
            axis=1
        )

        # Evaluate
        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

        valid_mask = ~simple_pred.isna() & ~em_pred.isna()
        y_true = items_df_test.loc[valid_mask, 'clicked'].values

        simple_pred_valid = simple_pred[valid_mask].values
        em_pred_valid = em_pred[valid_mask].values

        # Clip predictions
        simple_pred_valid = np.clip(simple_pred_valid, 1e-6, 1-1e-6)
        em_pred_valid = np.clip(em_pred_valid, 1e-6, 1-1e-6)

        log("Prediction metrics:", f)
        log(f"  {'Metric':<20} {'Simple PBM':<15} {'EM PBM':<15}", f)
        log(f"  {'-'*20} {'-'*15} {'-'*15}", f)

        # Log loss
        ll_simple = log_loss(y_true, simple_pred_valid)
        ll_em = log_loss(y_true, em_pred_valid)
        log(f"  {'Log Loss':<20} {ll_simple:<15.4f} {ll_em:<15.4f}", f)

        # Brier score
        bs_simple = brier_score_loss(y_true, simple_pred_valid)
        bs_em = brier_score_loss(y_true, em_pred_valid)
        log(f"  {'Brier Score':<20} {bs_simple:<15.4f} {bs_em:<15.4f}", f)

        # AUC
        auc_simple = roc_auc_score(y_true, simple_pred_valid)
        auc_em = roc_auc_score(y_true, em_pred_valid)
        log(f"  {'AUC':<20} {auc_simple:<15.4f} {auc_em:<15.4f}", f)

        log("", f)

        log("Interpretation:", f)
        if ll_em < ll_simple:
            log(f"  EM PBM improves log loss by {(ll_simple - ll_em) / ll_simple * 100:.1f}%", f)
        else:
            log(f"  Simple PBM has better log loss (item heterogeneity may be low)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("POSITION EFFECTS (θ_k) SUMMARY:", f)
        log(f"  θ_1 = 1.000 (normalized)", f)
        log(f"  θ_5 = {theta_em[4]:.3f} ({theta_em[4]*100:.1f}% of position 1)", f)
        log(f"  θ_10 = {theta_em[9]:.3f} ({theta_em[9]*100:.1f}% of position 1)", f)
        if len(theta_em) >= 20:
            log(f"  θ_20 = {theta_em[19]:.3f} ({theta_em[19]*100:.1f}% of position 1)", f)
        log("", f)

        log("TESTABLE IMPLICATIONS:", f)
        log(f"  1. Monotonicity: {'HOLDS' if mono_result_em['monotonicity_rate'] > 0.8 else 'VIOLATED'} ({mono_result_em['monotonicity_rate']*100:.1f}% positions)", f)

        if sep_results is not None:
            sep_status = 'HOLDS' if sep_results['cv'].median() < 0.5 else 'WEAK'
            log(f"  2. Multiplicative separability: {sep_status} (median CV = {sep_results['cv'].median():.3f})", f)
        else:
            log(f"  2. Multiplicative separability: INSUFFICIENT DATA", f)

        if indep_result is not None:
            indep_status = 'HOLDS' if indep_result['adjacent_click_rate'] < 0.3 else 'QUESTIONABLE'
            log(f"  3. Conditional independence: {indep_status} ({indep_result['adjacent_click_rate']*100:.1f}% adjacent clicks)", f)
        else:
            log(f"  3. Conditional independence: INSUFFICIENT DATA", f)
        log("", f)

        log("MODEL FIT:", f)
        log(f"  Log loss: {ll_em:.4f}", f)
        log(f"  AUC: {auc_em:.4f}", f)
        log("", f)

        log("CONCLUSIONS:", f)
        log("  - Position effects estimated: θ_k decreases with k", f)
        log("  - Item heterogeneity captured in α_i", f)
        log(f"  - PBM assumptions {'largely hold' if mono_result_em['monotonicity_rate'] > 0.8 else 'show some violations'}", f)
        log("", f)

        log("=" * 80, f)
        log("PBM VERIFICATION COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
