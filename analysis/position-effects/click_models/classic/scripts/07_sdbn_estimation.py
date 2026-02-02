#!/usr/bin/env python3
"""
Simplified Dynamic Bayesian Network (SDBN) Click Model

Mathematical Framework:
- Same as DBN but with constraint: σ_i × γ = 0
- This means: if user clicks, they always stop (no continuation)
- Simplifies to cascade-like model but with item-specific attractiveness

Model:
  P(C_k=1 | E_k=1) = α_i       (attractiveness)
  P(E_{k+1}=1 | C_k=1) = 0     (stop after click)
  P(E_{k+1}=1 | C_k=0) = γ_k   (continue if no click, position-dependent)

This is equivalent to:
  - User examines positions in order
  - At each position, clicks with probability α_i if examined
  - Click ends the session (satisfying click assumption)

Estimation:
  - MLE (no EM needed due to simplified structure)
  - γ_k estimated from continuation patterns

Reference: Simplified DBN is discussed in various click model surveys
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "07_sdbn_estimation.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# SDBN MODEL
# =============================================================================

class SimplifiedDBN:
    """
    Simplified DBN Click Model

    Key simplification: No continuation after click (σγ = 0)

    This means:
    - Click at position k implies examination at k and stop after k
    - No click at position k means either not examined or examined but not clicked

    Model:
      P(E_1) = 1
      P(C_k | E_k) = α_i
      P(E_{k+1} | C_k=0, E_k=1) = γ_k (position-specific continuation)
      P(E_{k+1} | C_k=1) = 0 (stop after click)

    Estimation via MLE:
      - For sessions with 0 clicks: all positions examined, no clicks
      - For sessions with 1 click at k: positions 1..k examined, click at k
      - For sessions with 2+ clicks: SDBN violated (skip or use last click)
    """

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions

        self.alpha = None   # Item attractiveness
        self.gamma = None   # Position-specific continuation
        self.fitted = False

    def fit_mle(self, sessions_df, verbose=True):
        """
        MLE estimation for SDBN.

        For SDBN, the likelihood factors cleanly:
        - P(session | α, γ) = product of position probabilities
        """
        if verbose:
            print("  Preparing data for SDBN estimation...")

        # Build item index
        all_items = set()
        for items in sessions_df['items']:
            all_items.update(items[:self.max_position])

        # Count item occurrences
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        if verbose:
            print(f"    Valid items: {n_items:,}")

        # Filter to single-click or no-click sessions (SDBN assumption)
        valid_sessions = sessions_df[sessions_df['n_clicks'] <= 1].copy()
        if verbose:
            print(f"    Sessions with 0-1 clicks: {len(valid_sessions):,} / {len(sessions_df):,}")

        # Also include multi-click sessions using first click only
        multi_click = sessions_df[sessions_df['n_clicks'] > 1].copy()
        if verbose:
            print(f"    Multi-click sessions (using first click): {len(multi_click):,}")

        # Process sessions
        # For each item: count impressions and clicks
        item_impressions = np.zeros(n_items)
        item_clicks = np.zeros(n_items)

        # For position-specific continuation
        position_continues = np.zeros(self.max_position)  # Continue to k+1 given no click at k
        position_total = np.zeros(self.max_position)       # At position k with no click

        for _, row in tqdm(valid_sessions.iterrows(), total=len(valid_sessions), desc="    Processing", disable=not verbose):
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]

            # Find first click position (or end of session)
            click_pos = None
            for k, (item, click) in enumerate(zip(items, clicks)):
                if item not in item_to_idx:
                    continue

                idx = item_to_idx[item]

                if click == 1:
                    click_pos = k
                    item_impressions[idx] += 1
                    item_clicks[idx] += 1
                    break
                else:
                    # No click at this position
                    item_impressions[idx] += 1

                    # Track continuation
                    if k < self.max_position - 1 and k < len(items) - 1:
                        position_total[k] += 1
                        # User continued to next position
                        position_continues[k] += 1

        # Also process multi-click sessions (use first click)
        for _, row in tqdm(multi_click.iterrows(), total=len(multi_click), desc="    Multi-click", disable=not verbose):
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]

            # Find first click
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

        # Estimate alpha (item attractiveness)
        alpha = np.zeros(n_items)
        for i in range(n_items):
            if item_impressions[i] > 0:
                alpha[i] = item_clicks[i] / item_impressions[i]
            else:
                alpha[i] = 0.01

        alpha = np.clip(alpha, 0.001, 0.999)

        # Estimate gamma (position continuation)
        gamma = np.ones(self.max_position)
        for k in range(self.max_position):
            if position_total[k] > 0:
                gamma[k] = position_continues[k] / position_total[k]

        gamma = np.clip(gamma, 0.01, 1.0)

        # Compute position effects (theta_k)
        theta = np.ones(self.max_position)
        theta[0] = 1.0
        for k in range(1, self.max_position):
            # P(examine k) = P(examine k-1) * (1 - P(click at k-1)) * gamma_{k-1}
            avg_alpha = np.mean(alpha)
            theta[k] = theta[k-1] * (1 - avg_alpha) * gamma[k-1]

        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.n_items = n_items
        self.fitted = True

        if verbose:
            print("    Estimation complete")

        return self

    def predict_ctr(self, position, product_id=None):
        """Predict CTR at position for item."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        if position > self.max_position:
            return 0.0

        theta_k = self.theta[position - 1]

        if product_id is None:
            return theta_k * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            return theta_k * self.alpha[self.item_to_idx[product_id]]
        else:
            return theta_k * np.mean(self.alpha)

    def get_position_effects(self):
        """Return position effects (theta)."""
        return self.theta.copy()

    def get_continuation_probs(self):
        """Return position-specific continuation probabilities (gamma)."""
        return self.gamma.copy()

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


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("SIMPLIFIED DBN (SDBN) CLICK MODEL", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  SDBN constraint: σ × γ = 0 (no continuation after click)", f)
        log("", f)
        log("  P(E_1) = 1                              (examine first)", f)
        log("  P(C_k | E_k) = α_i                      (attractiveness)", f)
        log("  P(E_{k+1} | C_k=0, E_k=1) = γ_k         (continue if no click)", f)
        log("  P(E_{k+1} | C_k=1) = 0                  (stop after click)", f)
        log("", f)
        log("ASSUMPTION: Click terminates session (single-click model)", f)
        log("ADVANTAGE: MLE estimation, no EM needed", f)
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

        # Session click statistics
        click_dist = sessions_df['n_clicks'].value_counts().sort_index()
        log("Session click distribution:", f)
        for n_clicks, count in click_dist.head(6).items():
            pct = count / len(sessions_df) * 100
            log(f"  {n_clicks} clicks: {count:,} ({pct:.1f}%)", f)
        log("", f)

        single_or_no_click = (sessions_df['n_clicks'] <= 1).sum()
        log(f"Sessions with 0-1 clicks: {single_or_no_click:,} ({single_or_no_click/len(sessions_df)*100:.1f}%)", f)
        log("  (These sessions satisfy SDBN assumption)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: SDBN Estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: SDBN ESTIMATION VIA MLE", f)
        log("-" * 40, f)
        log("", f)

        sdbn = SimplifiedDBN(max_position=20, min_impressions=5)
        log("Fitting SDBN via MLE...", f)
        sdbn.fit_mle(sessions_df, verbose=True)
        log("", f)

        # Item attractiveness
        alpha_stats = sdbn.get_alpha_stats()
        log("Item attractiveness (α_i) statistics:", f)
        log(f"  N items: {alpha_stats['n_items']:,}", f)
        log(f"  Mean: {alpha_stats['mean']:.4f}", f)
        log(f"  Std: {alpha_stats['std']:.4f}", f)
        log(f"  Min: {alpha_stats['min']:.4f}", f)
        log(f"  Max: {alpha_stats['max']:.4f}", f)
        log(f"  Median: {alpha_stats['median']:.4f}", f)
        log("", f)

        # Continuation probabilities
        gamma = sdbn.get_continuation_probs()
        log("Position-specific continuation (γ_k):", f)
        log(f"  {'Position':<10} {'γ_k':<12} {'1-γ_k (drop)':<12}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*12}", f)
        for k in range(min(15, len(gamma))):
            log(f"  {k+1:<10} {gamma[k]:<12.4f} {1-gamma[k]:<12.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Position Effects
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: POSITION EFFECTS (θ_k)", f)
        log("-" * 40, f)
        log("", f)

        theta_sdbn = sdbn.get_position_effects()

        # Compare with simple empirical
        ctr_by_pos = items_df[items_df['position'] <= 20].groupby('position')['clicked'].agg(['sum', 'count', 'mean']).reset_index()
        ctr_by_pos.columns = ['position', 'clicks', 'impressions', 'ctr']
        ctr_1 = ctr_by_pos[ctr_by_pos['position'] == 1]['ctr'].values[0]
        ctr_by_pos['theta_simple'] = ctr_by_pos['ctr'] / ctr_1

        log("Position effects comparison:", f)
        log(f"  {'Position':<10} {'θ (SDBN)':<12} {'θ (Simple)':<12} {'Diff':<10}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}", f)

        for k in range(min(20, len(theta_sdbn))):
            simple_theta = ctr_by_pos[ctr_by_pos['position'] == k+1]['theta_simple'].values
            simple_val = simple_theta[0] if len(simple_theta) > 0 else np.nan
            diff = theta_sdbn[k] - simple_val if not np.isnan(simple_val) else np.nan
            log(f"  {k+1:<10} {theta_sdbn[k]:<12.4f} {simple_val:<12.3f} {diff:<10.3f}", f)
        log("", f)

        # Monotonicity check
        violations = 0
        for k in range(len(theta_sdbn) - 1):
            if theta_sdbn[k] < theta_sdbn[k+1]:
                violations += 1

        log(f"Monotonicity: {len(theta_sdbn)-1-violations}/{len(theta_sdbn)-1} positions ({(1-violations/(len(theta_sdbn)-1))*100:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Cascade Assumption Validation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: CASCADE ASSUMPTION VALIDATION", f)
        log("-" * 40, f)
        log("", f)

        log("SDBN assumes: click terminates session (single-click)", f)
        log("", f)

        # How well does this assumption hold?
        multi_click_rate = (sessions_df['n_clicks'] > 1).mean()
        log(f"Multi-click rate (SDBN violation): {multi_click_rate*100:.2f}%", f)
        log("", f)

        # For multi-click sessions, where are subsequent clicks?
        multi_click_sessions = sessions_df[sessions_df['n_clicks'] > 1]

        if len(multi_click_sessions) > 0:
            log("Multi-click session analysis:", f)
            log(f"  N sessions: {len(multi_click_sessions):,}", f)

            gaps = []
            for _, row in multi_click_sessions.iterrows():
                clicks = row['clicks']
                click_pos = [i for i, c in enumerate(clicks) if c == 1]
                if len(click_pos) >= 2:
                    for j in range(len(click_pos) - 1):
                        gaps.append(click_pos[j+1] - click_pos[j])

            if gaps:
                log(f"  Mean gap between clicks: {np.mean(gaps):.2f}", f)
                log(f"  Median gap: {np.median(gaps):.0f}", f)

                # Proportion of adjacent clicks
                adjacent = sum(1 for g in gaps if g == 1) / len(gaps)
                log(f"  Adjacent clicks (gap=1): {adjacent*100:.1f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Prediction Quality
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: PREDICTION QUALITY", f)
        log("-" * 40, f)
        log("", f)

        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

        items_test = items_df[items_df['position'] <= 20].copy()

        # SDBN predictions
        sdbn_pred = items_test.apply(
            lambda row: sdbn.predict_ctr(int(row['position']), row['product_id']),
            axis=1
        )

        # Simple baseline
        simple_pred = items_test['position'].map(
            lambda p: ctr_by_pos[ctr_by_pos['position'] == p]['ctr'].values[0]
            if p in ctr_by_pos['position'].values else np.nan
        )

        valid_mask = ~sdbn_pred.isna() & ~simple_pred.isna()
        y_true = items_test.loc[valid_mask, 'clicked'].values

        sdbn_pred_valid = np.clip(sdbn_pred[valid_mask].values, 1e-6, 1-1e-6)
        simple_pred_valid = np.clip(simple_pred[valid_mask].values, 1e-6, 1-1e-6)

        log("Prediction metrics:", f)
        log(f"  {'Metric':<20} {'Simple':<15} {'SDBN':<15}", f)
        log(f"  {'-'*20} {'-'*15} {'-'*15}", f)

        ll_simple = log_loss(y_true, simple_pred_valid)
        ll_sdbn = log_loss(y_true, sdbn_pred_valid)
        log(f"  {'Log Loss':<20} {ll_simple:<15.4f} {ll_sdbn:<15.4f}", f)

        bs_simple = brier_score_loss(y_true, simple_pred_valid)
        bs_sdbn = brier_score_loss(y_true, sdbn_pred_valid)
        log(f"  {'Brier Score':<20} {bs_simple:<15.4f} {bs_sdbn:<15.4f}", f)

        auc_simple = roc_auc_score(y_true, simple_pred_valid)
        auc_sdbn = roc_auc_score(y_true, sdbn_pred_valid)
        log(f"  {'AUC':<20} {auc_simple:<15.4f} {auc_sdbn:<15.4f}", f)

        log("", f)

        # Improvement
        ll_improvement = (ll_simple - ll_sdbn) / ll_simple * 100
        log(f"Log loss improvement over simple: {ll_improvement:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Stratified Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: STRATIFIED ANALYSIS BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        for placement in sorted(sessions_df['placement'].unique()):
            placement_sessions = sessions_df[sessions_df['placement'] == placement]

            if len(placement_sessions) < 500:
                continue

            log(f"PLACEMENT {placement}:", f)
            log(f"  Sessions: {len(placement_sessions):,}", f)

            # Fit SDBN for this placement
            try:
                sdbn_placement = SimplifiedDBN(max_position=10, min_impressions=3)
                sdbn_placement.fit_mle(placement_sessions, verbose=False)

                theta_p = sdbn_placement.get_position_effects()
                gamma_p = sdbn_placement.get_continuation_probs()

                log(f"  Mean α: {np.mean(sdbn_placement.alpha):.4f}", f)
                log(f"  θ_5: {theta_p[4]:.3f}", f)
                log(f"  Mean γ: {np.mean(gamma_p[:5]):.3f}", f)
            except Exception as e:
                log(f"  Could not fit: {str(e)}", f)

            log("", f)

        # -----------------------------------------------------------------
        # Section 7: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("SDBN PARAMETERS:", f)
        log(f"  Mean α (attractiveness): {alpha_stats['mean']:.4f}", f)
        log(f"  Mean γ (continuation): {np.mean(gamma[:10]):.4f}", f)
        log("", f)

        log("POSITION EFFECTS (θ_k):", f)
        log(f"  θ_1 = 1.0000 (normalized)", f)
        log(f"  θ_5 = {theta_sdbn[4]:.4f} ({theta_sdbn[4]*100:.1f}% of position 1)", f)
        log(f"  θ_10 = {theta_sdbn[9]:.4f} ({theta_sdbn[9]*100:.1f}% of position 1)", f)
        if len(theta_sdbn) >= 20:
            log(f"  θ_20 = {theta_sdbn[19]:.4f} ({theta_sdbn[19]*100:.1f}% of position 1)", f)
        log("", f)

        log("MODEL FIT:", f)
        log(f"  Log loss: {ll_sdbn:.4f}", f)
        log(f"  AUC: {auc_sdbn:.4f}", f)
        log(f"  LL improvement over simple: {ll_improvement:.2f}%", f)
        log("", f)

        log("KEY FINDINGS:", f)
        log(f"  1. Cascade assumption violated in {multi_click_rate*100:.1f}% of sessions", f)
        log(f"  2. Position effect decay: θ_10 = {theta_sdbn[9]:.2f}x position 1", f)
        log(f"  3. Mean continuation probability: {np.mean(gamma[:10]):.3f}", f)
        log("", f)

        log("COMPARISON WITH DBN:", f)
        log("  - SDBN uses MLE (faster than EM)", f)
        log("  - SDBN ignores multi-click information", f)
        log("  - SDBN position effects are mechanically derived from γ", f)
        log("", f)

        log("=" * 80, f)
        log("SDBN ESTIMATION COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
