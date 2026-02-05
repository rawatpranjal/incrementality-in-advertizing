#!/usr/bin/env python3
"""
Dynamic Bayesian Network (DBN) Click Model Estimation

Mathematical Framework:
- Examination: E_k = 1 if position k is examined
- Click: C_k = 1 if item at k is clicked (given examined)
- Satisfaction: S_k = 1 if click satisfied user

Model:
  P(C_k=1 | E_k=1) = α_i       (attractiveness)
  P(S_k=1 | C_k=1) = σ_i       (satisfaction)
  P(E_{k+1}=1 | S_k=0) = γ     (continuation)
  P(E_1=1) = 1                 (always examine first)

Key insight: DBN handles multi-click by allowing continuation after unsatisfying clicks.

Estimation via EM:
- E-step: Compute expected examination given observed clicks
- M-step: Update α, σ, γ

Reference: Chapelle & Zhang (2009) "A Dynamic Bayesian Network Click Model for Web Search Ranking"
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
OUTPUT_FILE = RESULTS_DIR / "06_dbn_estimation.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# DBN MODEL
# =============================================================================

class DynamicBayesianNetwork:
    """
    Dynamic Bayesian Network (DBN) Click Model

    Generative process:
    1. User examines position 1 (E_1 = 1)
    2. If examined, user clicks with probability α_i
    3. If clicked, user is satisfied with probability σ_i
    4. If satisfied, user stops; otherwise continues with probability γ

    Parameters:
    - α_i: attractiveness of item i
    - σ_i: satisfaction probability for item i
    - γ: continuation probability after unsatisfied click
    """

    def __init__(self, max_position=20, min_impressions=5, use_item_sigma=False):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.use_item_sigma = use_item_sigma  # If True, σ varies by item; else pooled

        self.alpha = None   # Item attractiveness
        self.sigma = None   # Satisfaction (scalar or per-item)
        self.gamma = None   # Continuation probability
        self.fitted = False

    def fit_em(self, sessions_df, max_iter=50, tol=1e-5, verbose=True):
        """
        EM algorithm for DBN estimation.

        For each session, compute:
        - P(E_k | clicks) using forward-backward
        - Update parameters using expected counts
        """
        if verbose:
            print("  Preparing data for DBN estimation...")

        # Build item index
        all_items = set()
        for items in sessions_df['items']:
            all_items.update(items[:self.max_position])

        # Filter to items with sufficient data
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        if verbose:
            print(f"    Valid items: {n_items:,}")

        # Initialize parameters
        alpha = np.full(n_items, 0.03)  # Initial attractiveness ~3%
        sigma = 0.5  # Initial satisfaction probability
        gamma = 0.8  # Initial continuation probability

        # Preprocess sessions
        sessions_data = []
        for _, row in tqdm(sessions_df.iterrows(), total=len(sessions_df), desc="    Preprocessing", disable=not verbose):
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]

            # Map items to indices (skip unknown items)
            item_indices = []
            valid_positions = []
            valid_clicks = []

            for k, (item, click) in enumerate(zip(items, clicks)):
                if item in item_to_idx:
                    item_indices.append(item_to_idx[item])
                    valid_positions.append(k)
                    valid_clicks.append(click)

            if len(item_indices) > 0:
                sessions_data.append({
                    'item_indices': np.array(item_indices),
                    'positions': np.array(valid_positions),
                    'clicks': np.array(valid_clicks),
                    'n_positions': len(item_indices)
                })

        if verbose:
            print(f"    Valid sessions: {len(sessions_data):,}")

        ll_history = []

        for iteration in range(max_iter):
            # E-step: Compute expected examination probabilities
            # Using simplified approach: P(E_k=1 | clicks) via forward-backward

            # Accumulators for M-step
            alpha_num = np.zeros(n_items)   # Sum of clicks for each item
            alpha_denom = np.zeros(n_items) # Sum of expected examinations
            sigma_num = 0.0  # Sum of satisfied (stopped) after click
            sigma_denom = 0.0  # Sum of clicks
            gamma_num = 0.0  # Sum of continuations after click
            gamma_denom = 0.0  # Sum of unsatisfied clicks

            total_ll = 0.0

            for session in sessions_data:
                item_idx = session['item_indices']
                clicks = session['clicks']
                n_pos = session['n_positions']

                # Compute examination probabilities using forward algorithm
                # P(E_k=1) depends on previous clicks and satisfaction

                # Forward pass: P(E_k=1 | C_1, ..., C_{k-1})
                exam_prob = np.zeros(n_pos)
                exam_prob[0] = 1.0  # Always examine first position

                for k in range(1, n_pos):
                    # P(E_k) = sum over states at k-1
                    # If C_{k-1}=0: continue with prob 1 (didn't click)
                    # If C_{k-1}=1: continue with prob (1-σ)γ (clicked but unsatisfied)

                    prev_alpha = alpha[item_idx[k-1]]
                    prev_click = clicks[k-1]

                    if prev_click == 0:
                        # Didn't click at k-1, definitely continue
                        exam_prob[k] = exam_prob[k-1]
                    else:
                        # Clicked at k-1, continue if unsatisfied
                        exam_prob[k] = exam_prob[k-1] * (1 - sigma) * gamma

                # Compute session log-likelihood
                session_ll = 0.0
                for k in range(n_pos):
                    p_click = exam_prob[k] * alpha[item_idx[k]]
                    p_click = np.clip(p_click, 1e-10, 1-1e-10)

                    if clicks[k] == 1:
                        session_ll += np.log(p_click)
                    else:
                        session_ll += np.log(1 - p_click)

                total_ll += session_ll

                # Accumulate for M-step
                for k in range(n_pos):
                    alpha_num[item_idx[k]] += clicks[k]
                    alpha_denom[item_idx[k]] += exam_prob[k]

                    if clicks[k] == 1:
                        sigma_denom += 1

                        # Did user stop after this click?
                        if k == n_pos - 1:
                            # Last position - assume stopped (satisfied)
                            sigma_num += 1
                        else:
                            # Check if there are any more clicks after this
                            more_clicks = clicks[k+1:].sum()
                            if more_clicks == 0:
                                # No more clicks - could be satisfied or continued but didn't find anything
                                # Approximate: weight by probability of satisfaction
                                sigma_num += sigma  # Current estimate
                            else:
                                # More clicks - definitely unsatisfied
                                gamma_num += 1
                                gamma_denom += 1

            ll_history.append(total_ll)

            # M-step: Update parameters
            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                if alpha_denom[i] > 0:
                    new_alpha[i] = alpha_num[i] / alpha_denom[i]
                else:
                    new_alpha[i] = alpha[i]
            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            new_sigma = sigma_num / sigma_denom if sigma_denom > 0 else sigma
            new_sigma = np.clip(new_sigma, 0.01, 0.99)

            new_gamma = gamma_num / gamma_denom if gamma_denom > 0 else gamma
            new_gamma = np.clip(new_gamma, 0.01, 0.99)

            # Check convergence
            alpha_change = np.max(np.abs(new_alpha - alpha))
            sigma_change = abs(new_sigma - sigma)
            gamma_change = abs(new_gamma - gamma)

            alpha = new_alpha
            sigma = new_sigma
            gamma = new_gamma

            if verbose and iteration % 5 == 0:
                print(f"    Iter {iteration}: LL={total_ll:.2f}, α_change={alpha_change:.6f}, σ={sigma:.3f}, γ={gamma:.3f}")

            if alpha_change < tol and sigma_change < tol and gamma_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.ll_history = ll_history
        self.n_items = n_items
        self.fitted = True

        return self

    def compute_examination_prob(self, position, n_clicks_before=0):
        """
        Compute examination probability at position k.

        θ_k in PBM terms, but depends on click history in DBN.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        if position == 1:
            return 1.0

        # Expected examination probability marginalizing over click patterns
        # Simplified: assume average click rate at each position
        avg_alpha = np.mean(self.alpha)

        exam_prob = 1.0
        for k in range(1, position):
            # At each position, either no click (continue) or click and unsatisfied (continue)
            p_no_click = 1 - exam_prob * avg_alpha
            p_click_continue = exam_prob * avg_alpha * (1 - self.sigma) * self.gamma
            exam_prob = p_no_click + p_click_continue

        return exam_prob

    def get_position_effects(self, max_k=20):
        """
        Compute position effects comparable to PBM θ_k.
        """
        theta = np.zeros(max_k)
        theta[0] = 1.0

        for k in range(1, max_k):
            theta[k] = self.compute_examination_prob(k + 1)

        return theta

    def predict_ctr(self, position, product_id=None):
        """Predict CTR at position for item."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        exam_prob = self.compute_examination_prob(position)

        if product_id is None:
            return exam_prob * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            return exam_prob * self.alpha[self.item_to_idx[product_id]]
        else:
            return exam_prob * np.mean(self.alpha)

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
        log("DYNAMIC BAYESIAN NETWORK (DBN) CLICK MODEL", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  P(C_k=1 | E_k=1) = α_i       (attractiveness)", f)
        log("  P(S_k=1 | C_k=1) = σ         (satisfaction)", f)
        log("  P(E_{k+1}=1 | S_k=0) = γ     (continuation after unsatisfied click)", f)
        log("  P(E_1=1) = 1                 (always examine first)", f)
        log("", f)
        log("KEY DIFFERENCE FROM PBM:", f)
        log("  - DBN allows continuation after click (multi-click handling)", f)
        log("  - Position effect depends on click history, not just position", f)
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

        # Multi-click statistics
        multi_click_rate = (sessions_df['n_clicks'] >= 2).mean()
        log(f"Multi-click sessions: {(sessions_df['n_clicks'] >= 2).sum():,} ({multi_click_rate*100:.1f}%)", f)
        log(f"Average clicks per session (with clicks): {sessions_df[sessions_df['n_clicks'] > 0]['n_clicks'].mean():.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: DBN Estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: DBN ESTIMATION VIA EM", f)
        log("-" * 40, f)
        log("", f)

        dbn = DynamicBayesianNetwork(max_position=20, min_impressions=5)
        log("Fitting DBN via EM...", f)
        dbn.fit_em(sessions_df, max_iter=50, verbose=True)
        log("", f)

        # Parameters
        log("ESTIMATED PARAMETERS:", f)
        log(f"  σ (satisfaction probability): {dbn.sigma:.4f}", f)
        log(f"  γ (continuation probability): {dbn.gamma:.4f}", f)
        log(f"  (1-σ)γ (prob continue after click): {(1-dbn.sigma)*dbn.gamma:.4f}", f)
        log("", f)

        alpha_stats = dbn.get_alpha_stats()
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
        log(f"  Initial LL: {dbn.ll_history[0]:.2f}", f)
        log(f"  Final LL: {dbn.ll_history[-1]:.2f}", f)
        log(f"  Iterations: {len(dbn.ll_history)}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Position Effects (θ_k)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: POSITION EFFECTS (θ_k)", f)
        log("-" * 40, f)
        log("", f)

        log("NOTE: In DBN, θ_k depends on click history. We compute marginal θ_k.", f)
        log("", f)

        theta_dbn = dbn.get_position_effects(max_k=20)

        # Compare with simple PBM
        ctr_by_pos = items_df[items_df['position'] <= 20].groupby('position')['clicked'].agg(['sum', 'count', 'mean']).reset_index()
        ctr_by_pos.columns = ['position', 'clicks', 'impressions', 'ctr']
        ctr_1 = ctr_by_pos[ctr_by_pos['position'] == 1]['ctr'].values[0]
        ctr_by_pos['theta_simple'] = ctr_by_pos['ctr'] / ctr_1

        log("Position effects comparison:", f)
        log(f"  {'Position':<10} {'θ (DBN)':<12} {'θ (Simple)':<12} {'Diff':<10} {'Raw CTR':<10}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", f)

        for k in range(20):
            simple_theta = ctr_by_pos[ctr_by_pos['position'] == k+1]['theta_simple'].values
            simple_val = simple_theta[0] if len(simple_theta) > 0 else np.nan
            raw_ctr = ctr_by_pos[ctr_by_pos['position'] == k+1]['ctr'].values
            raw_ctr_val = raw_ctr[0] * 100 if len(raw_ctr) > 0 else np.nan
            diff = theta_dbn[k] - simple_val if not np.isnan(simple_val) else np.nan
            log(f"  {k+1:<10} {theta_dbn[k]:<12.3f} {simple_val:<12.3f} {diff:<10.3f} {raw_ctr_val:<10.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Multi-Click Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: MULTI-CLICK ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("OBJECTIVE: Validate DBN's multi-click handling", f)
        log("", f)

        # Observed vs predicted multi-click rate
        # Under DBN: P(2+ clicks) = P(click_1) * (1-σ) * γ * P(click_2 | E_2) * ...

        observed_multi_click = (sessions_df['n_clicks'] >= 2).mean()

        # Predicted: approximate
        avg_alpha = np.mean(dbn.alpha)
        p_click_1 = avg_alpha
        p_continue = (1 - dbn.sigma) * dbn.gamma
        p_click_2_given_continue = avg_alpha
        predicted_multi_click = p_click_1 * p_continue * p_click_2_given_continue

        log("Multi-click prediction:", f)
        log(f"  Observed multi-click rate: {observed_multi_click*100:.2f}%", f)
        log(f"  Predicted (approx): {predicted_multi_click*100:.2f}%", f)
        log("", f)

        # Click position analysis in multi-click sessions
        multi_click_sessions = sessions_df[sessions_df['n_clicks'] >= 2].copy()

        if len(multi_click_sessions) > 0:
            first_click_positions = []
            second_click_positions = []
            click_gaps = []

            for _, session in multi_click_sessions.iterrows():
                clicks = session['clicks']
                click_pos = [i for i, c in enumerate(clicks) if c == 1]
                if len(click_pos) >= 2:
                    first_click_positions.append(click_pos[0] + 1)
                    second_click_positions.append(click_pos[1] + 1)
                    click_gaps.append(click_pos[1] - click_pos[0])

            log("Click patterns in multi-click sessions:", f)
            log(f"  Mean first click position: {np.mean(first_click_positions):.2f}", f)
            log(f"  Mean second click position: {np.mean(second_click_positions):.2f}", f)
            log(f"  Mean gap between clicks: {np.mean(click_gaps):.2f}", f)
            log("", f)

            # Gap distribution
            log("Gap between consecutive clicks:", f)
            gap_counts = pd.Series(click_gaps).value_counts().sort_index()
            for gap, count in gap_counts.head(10).items():
                log(f"    Gap {gap}: {count} ({count/len(click_gaps)*100:.1f}%)", f)
            log("", f)

        # -----------------------------------------------------------------
        # Section 5: Satisfaction Parameter Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: SATISFACTION PARAMETER ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log(f"Estimated σ (satisfaction): {dbn.sigma:.4f}", f)
        log(f"Estimated γ (continuation): {dbn.gamma:.4f}", f)
        log("", f)

        log("INTERPRETATION:", f)
        log(f"  - When a user clicks, they are satisfied {dbn.sigma*100:.1f}% of the time", f)
        log(f"  - If unsatisfied, they continue examining with {dbn.gamma*100:.1f}% probability", f)
        log(f"  - Net continuation rate after click: {(1-dbn.sigma)*dbn.gamma*100:.1f}%", f)
        log("", f)

        # Implied examination decay
        log("Implied examination decay (marginal):", f)
        for k in [1, 5, 10, 15, 20]:
            if k <= len(theta_dbn):
                log(f"  Position {k}: θ = {theta_dbn[k-1]:.3f} ({theta_dbn[k-1]*100:.1f}% of position 1)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Prediction Quality
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: PREDICTION QUALITY", f)
        log("-" * 40, f)
        log("", f)

        from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

        # Predict CTR for each item
        items_test = items_df[items_df['position'] <= 20].copy()

        dbn_pred = items_test.apply(
            lambda row: dbn.predict_ctr(int(row['position']), row['product_id']),
            axis=1
        )

        # Simple baseline
        simple_pred = items_test['position'].map(
            lambda p: ctr_by_pos[ctr_by_pos['position'] == p]['ctr'].values[0]
            if p in ctr_by_pos['position'].values else np.nan
        )

        valid_mask = ~dbn_pred.isna() & ~simple_pred.isna()
        y_true = items_test.loc[valid_mask, 'clicked'].values

        dbn_pred_valid = np.clip(dbn_pred[valid_mask].values, 1e-6, 1-1e-6)
        simple_pred_valid = np.clip(simple_pred[valid_mask].values, 1e-6, 1-1e-6)

        log("Prediction metrics:", f)
        log(f"  {'Metric':<20} {'Simple':<15} {'DBN':<15}", f)
        log(f"  {'-'*20} {'-'*15} {'-'*15}", f)

        ll_simple = log_loss(y_true, simple_pred_valid)
        ll_dbn = log_loss(y_true, dbn_pred_valid)
        log(f"  {'Log Loss':<20} {ll_simple:<15.4f} {ll_dbn:<15.4f}", f)

        bs_simple = brier_score_loss(y_true, simple_pred_valid)
        bs_dbn = brier_score_loss(y_true, dbn_pred_valid)
        log(f"  {'Brier Score':<20} {bs_simple:<15.4f} {bs_dbn:<15.4f}", f)

        auc_simple = roc_auc_score(y_true, simple_pred_valid)
        auc_dbn = roc_auc_score(y_true, dbn_pred_valid)
        log(f"  {'AUC':<20} {auc_simple:<15.4f} {auc_dbn:<15.4f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Stratified Analysis by Placement
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: STRATIFIED ANALYSIS BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        for placement in sorted(sessions_df['placement'].unique()):
            placement_sessions = sessions_df[sessions_df['placement'] == placement]

            if len(placement_sessions) < 500:
                continue

            log(f"PLACEMENT {placement}:", f)
            log(f"  Sessions: {len(placement_sessions):,}", f)
            log(f"  Multi-click rate: {(placement_sessions['n_clicks'] >= 2).mean()*100:.1f}%", f)

            # Fit DBN for this placement
            try:
                dbn_placement = DynamicBayesianNetwork(max_position=10, min_impressions=3)
                dbn_placement.fit_em(placement_sessions, max_iter=30, verbose=False)

                log(f"  σ (satisfaction): {dbn_placement.sigma:.3f}", f)
                log(f"  γ (continuation): {dbn_placement.gamma:.3f}", f)
                log(f"  Mean α: {np.mean(dbn_placement.alpha):.4f}", f)
            except Exception as e:
                log(f"  Could not fit: {str(e)}", f)

            log("", f)

        # -----------------------------------------------------------------
        # Section 8: Model Comparison Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("DBN PARAMETERS:", f)
        log(f"  σ (satisfaction): {dbn.sigma:.4f}", f)
        log(f"  γ (continuation): {dbn.gamma:.4f}", f)
        log(f"  Mean α (attractiveness): {np.mean(dbn.alpha):.4f}", f)
        log("", f)

        log("POSITION EFFECTS (θ_k):", f)
        log(f"  θ_1 = 1.000 (normalized)", f)
        log(f"  θ_5 = {theta_dbn[4]:.3f} ({theta_dbn[4]*100:.1f}% of position 1)", f)
        log(f"  θ_10 = {theta_dbn[9]:.3f} ({theta_dbn[9]*100:.1f}% of position 1)", f)
        if len(theta_dbn) >= 20:
            log(f"  θ_20 = {theta_dbn[19]:.3f} ({theta_dbn[19]*100:.1f}% of position 1)", f)
        log("", f)

        log("MODEL FIT:", f)
        log(f"  Log loss: {ll_dbn:.4f}", f)
        log(f"  AUC: {auc_dbn:.4f}", f)
        log(f"  LL improvement over simple: {(ll_simple - ll_dbn) / ll_simple * 100:.2f}%", f)
        log("", f)

        log("KEY FINDINGS:", f)
        log(f"  1. Multi-click rate ({multi_click_rate*100:.1f}%) suggests sequential behavior", f)
        log(f"  2. Satisfaction σ={dbn.sigma:.2f} indicates {dbn.sigma*100:.0f}% of clicks end session", f)
        log(f"  3. Position effect decay is {'gradual' if theta_dbn[9] > 0.5 else 'steep'}", f)
        log("", f)

        log("=" * 80, f)
        log("DBN ESTIMATION COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
