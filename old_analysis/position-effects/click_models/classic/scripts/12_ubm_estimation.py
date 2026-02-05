#!/usr/bin/env python3
"""
User Browsing Model (UBM) Click Model Estimation

Mathematical Framework:
- Examination hypothesis: P(C_r = 1) = α_u × γ_{r,r'}
- α_u = attractiveness of document u
- γ_{r,r'} = examination probability given last click was at position r'
- γ_{r,0} = γ_r (no previous click case)

Key difference from PBM: examination depends on distance to last click, not just position.
Key difference from DBN: explicit modeling of examination given any prior click position.

Estimation via EM:
- E-step: Compute P(E_r = 1 | clicks) for each position
- M-step: Update α_u, γ_{r,r'}

Reference: Chuklin et al. (2015) "Click Models for Web Search" Chapter 5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "12_ubm_estimation.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# UBM MODEL
# =============================================================================

class UserBrowsingModel:
    """
    User Browsing Model (UBM) for click prediction.

    Model: P(C_r = 1 | last_click = r') = α_u × γ_{r,r'}

    Parameters:
    - α_u ∈ [0,1]: attractiveness of product u
    - γ_{r,r'} ∈ [0,1]: P(examine rank r | last click at r')
      - r' = 0: no prior click in session
      - r' > 0: last click was at position r'

    Key insight: examination probability depends on distance from last click,
    capturing "restart" behavior after clicking.
    """

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions

        self.alpha = None   # Item attractiveness
        self.gamma = None   # Examination: shape (max_position, max_position+1)
        self.fitted = False

    def fit_em(self, sessions_df, max_iter=50, tol=1e-5, verbose=True):
        """
        EM algorithm for UBM estimation.

        E-step: Compute expected examination P(E_r=1 | clicks)
        M-step: Update α, γ
        """
        if verbose:
            print("  Preparing data for UBM estimation...")

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

        # Preprocess sessions
        sessions_data = []
        for _, row in tqdm(sessions_df.iterrows(), total=len(sessions_df), desc="    Preprocessing", disable=not verbose):
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]

            # Map items to indices
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

        # Initialize parameters
        # α: item attractiveness - initialize from empirical CTR per item
        item_clicks = np.zeros(n_items)
        item_impr = np.zeros(n_items)
        for session in sessions_data:
            for idx, click in zip(session['item_indices'], session['clicks']):
                item_impr[idx] += 1
                item_clicks[idx] += click

        alpha = np.zeros(n_items)
        for i in range(n_items):
            if item_impr[i] > 0:
                alpha[i] = np.clip(item_clicks[i] / item_impr[i] + 0.01, 0.001, 0.999)
            else:
                alpha[i] = 0.03

        # γ: examination probabilities - shape (max_position, max_position+1)
        # γ[r, r'] = P(examine r | last click at r')
        # r' = 0 means no previous click
        gamma = np.zeros((self.max_position, self.max_position + 1))

        # Initialize gamma with reasonable priors
        for r in range(self.max_position):
            # No prior click: decay with position
            gamma[r, 0] = max(0.1, 1.0 - 0.03 * r)

            # Prior click at r': depends on distance
            for r_prime in range(1, self.max_position + 1):
                if r_prime - 1 < r:
                    # r' is before r: examination likely if close
                    distance = r - (r_prime - 1)
                    gamma[r, r_prime] = max(0.1, 1.0 - 0.05 * distance)
                else:
                    # r' is at or after r: already passed, low prob
                    gamma[r, r_prime] = 0.1

        ll_history = []

        for iteration in range(max_iter):
            # E-step: Compute expected examination probabilities
            # For each position in each session

            # Accumulators for M-step
            alpha_num = np.zeros(n_items)    # Sum of clicks
            alpha_denom = np.zeros(n_items)  # Sum of expected examinations
            gamma_num = np.zeros((self.max_position, self.max_position + 1))
            gamma_denom = np.zeros((self.max_position, self.max_position + 1))

            total_ll = 0.0

            for session in sessions_data:
                item_idx = session['item_indices']
                positions = session['positions']
                clicks = session['clicks']
                n_pos = session['n_positions']

                # Track last click position (0 = no click yet, 1-indexed for actual clicks)
                last_click = 0  # 0-indexed internally, but gamma uses 1-indexed

                exam_probs = np.zeros(n_pos)
                session_ll = 0.0

                for k in range(n_pos):
                    pos = positions[k]  # 0-indexed position
                    if pos >= self.max_position:
                        continue

                    # Get examination probability given last click
                    # gamma index: [position, last_click_position+1]
                    # last_click = 0 means no prior click, maps to gamma[:, 0]
                    # last_click = j means last click at position j-1 (0-indexed), maps to gamma[:, j]
                    exam_prob = gamma[pos, last_click]

                    # P(click) = P(examined) * P(click | examined)
                    a_u = alpha[item_idx[k]]
                    p_click = exam_prob * a_u
                    p_click = np.clip(p_click, 1e-10, 1 - 1e-10)

                    # Compute examination probability given click observation
                    if clicks[k] == 1:
                        # Clicked implies examined
                        exam_given_obs = 1.0
                        session_ll += np.log(p_click)
                    else:
                        # Not clicked: could be not examined or examined but not attractive
                        # P(E=1 | C=0) = P(E=1, C=0) / P(C=0)
                        # P(E=1, C=0) = P(E=1) * (1 - α)
                        # P(C=0) = P(E=0) + P(E=1)(1-α) = (1-γ) + γ(1-α)
                        p_not_click = 1 - p_click
                        p_exam_no_click = exam_prob * (1 - a_u)
                        if p_not_click > 1e-10:
                            exam_given_obs = p_exam_no_click / p_not_click
                        else:
                            exam_given_obs = exam_prob
                        session_ll += np.log(1 - p_click)

                    exam_probs[k] = exam_given_obs

                    # Accumulate for M-step
                    alpha_num[item_idx[k]] += clicks[k]
                    alpha_denom[item_idx[k]] += exam_given_obs

                    gamma_num[pos, last_click] += exam_given_obs
                    gamma_denom[pos, last_click] += 1

                    # Update last click if clicked
                    if clicks[k] == 1:
                        last_click = pos + 1  # 1-indexed for gamma

                total_ll += session_ll

            ll_history.append(total_ll)

            # M-step: Update parameters
            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                if alpha_denom[i] > 1e-10:
                    new_alpha[i] = alpha_num[i] / alpha_denom[i]
                else:
                    new_alpha[i] = alpha[i]
            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            new_gamma = np.zeros_like(gamma)
            for r in range(self.max_position):
                for r_prime in range(self.max_position + 1):
                    if gamma_denom[r, r_prime] > 1e-10:
                        new_gamma[r, r_prime] = gamma_num[r, r_prime] / gamma_denom[r, r_prime]
                    else:
                        new_gamma[r, r_prime] = gamma[r, r_prime]
            new_gamma = np.clip(new_gamma, 0.01, 1.0)

            # Check convergence
            alpha_change = np.max(np.abs(new_alpha - alpha))
            gamma_change = np.max(np.abs(new_gamma - gamma))

            alpha = new_alpha
            gamma = new_gamma

            if verbose and iteration % 5 == 0:
                print(f"    Iter {iteration}: LL={total_ll:.2f}, α_change={alpha_change:.6f}, γ_change={gamma_change:.6f}")

            if alpha_change < tol and gamma_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

        self.alpha = alpha
        self.gamma = gamma
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.ll_history = ll_history
        self.n_items = n_items
        self.fitted = True

        return self

    def get_position_effects(self, last_click=0):
        """
        Return position effects for a given last_click context.

        last_click=0: no prior click (standard position decay)
        last_click=k: last click was at position k
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        return self.gamma[:, last_click].copy()

    def predict_ctr(self, position, product_id=None, last_click=0):
        """Predict CTR at position given last click context."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        if position > self.max_position or position < 1:
            return 0.0

        gamma_val = self.gamma[position - 1, last_click]

        if product_id is None:
            return gamma_val * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            return gamma_val * self.alpha[self.item_to_idx[product_id]]
        else:
            return gamma_val * np.mean(self.alpha)

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
        log("USER BROWSING MODEL (UBM) CLICK MODEL", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  P(C_r = 1 | last_click = r') = α_u × γ_{r,r'}", f)
        log("", f)
        log("  α_u = attractiveness of product u", f)
        log("  γ_{r,r'} = P(examine rank r | last click at r')", f)
        log("  γ_{r,0} = P(examine rank r | no prior click)", f)
        log("", f)
        log("KEY DIFFERENCE FROM PBM:", f)
        log("  - UBM: examination depends on DISTANCE to last click", f)
        log("  - PBM: examination depends only on absolute position", f)
        log("", f)
        log("KEY DIFFERENCE FROM DBN:", f)
        log("  - UBM: explicit γ_{r,r'} for any prior click position", f)
        log("  - DBN: binary satisfaction/continuation model", f)
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

        log("Session click distribution:", f)
        click_dist = sessions_df['n_clicks'].value_counts().sort_index()
        for n_clicks, count in click_dist.head(6).items():
            pct = count / len(sessions_df) * 100
            log(f"  {n_clicks} clicks: {count:,} ({pct:.1f}%)", f)
        log("", f)

        multi_click_rate = (sessions_df['n_clicks'] >= 2).mean()
        log(f"Multi-click sessions: {(sessions_df['n_clicks'] >= 2).sum():,} ({multi_click_rate*100:.1f}%)", f)
        log("  (UBM explicitly models examination after clicks)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: UBM Estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: UBM ESTIMATION VIA EM", f)
        log("-" * 40, f)
        log("", f)

        ubm = UserBrowsingModel(max_position=20, min_impressions=5)
        log("Fitting UBM via EM...", f)
        ubm.fit_em(sessions_df, max_iter=50, verbose=True)
        log("", f)

        # EM convergence
        log("EM convergence:", f)
        log(f"  Initial LL: {ubm.ll_history[0]:.2f}", f)
        log(f"  Final LL: {ubm.ll_history[-1]:.2f}", f)
        log(f"  Iterations: {len(ubm.ll_history)}", f)
        log(f"  LL improvement: {ubm.ll_history[-1] - ubm.ll_history[0]:.2f}", f)
        log("", f)

        # Item attractiveness
        alpha_stats = ubm.get_alpha_stats()
        log("Item attractiveness (α_u) statistics:", f)
        log(f"  N items: {alpha_stats['n_items']:,}", f)
        log(f"  Mean: {alpha_stats['mean']:.4f}", f)
        log(f"  Std: {alpha_stats['std']:.4f}", f)
        log(f"  Min: {alpha_stats['min']:.4f}", f)
        log(f"  Max: {alpha_stats['max']:.4f}", f)
        log(f"  Median: {alpha_stats['median']:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Position Effects (No Prior Click)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: POSITION EFFECTS (NO PRIOR CLICK)", f)
        log("-" * 40, f)
        log("", f)

        log("γ_{r,0}: Examination probability with no prior click", f)
        log("  (Comparable to PBM θ_k)", f)
        log("", f)

        theta_ubm = ubm.get_position_effects(last_click=0)

        # Compare with simple empirical
        ctr_by_pos = items_df[items_df['position'] <= 20].groupby('position')['clicked'].agg(['sum', 'count', 'mean']).reset_index()
        ctr_by_pos.columns = ['position', 'clicks', 'impressions', 'ctr']
        ctr_1 = ctr_by_pos[ctr_by_pos['position'] == 1]['ctr'].values[0]
        ctr_by_pos['theta_simple'] = ctr_by_pos['ctr'] / ctr_1

        log("Position effects comparison:", f)
        log(f"  {'Position':<10} {'γ_{r,0} (UBM)':<15} {'θ (Simple)':<12} {'Diff':<10} {'Raw CTR':<10}", f)
        log(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*10} {'-'*10}", f)

        for k in range(min(20, len(theta_ubm))):
            simple_theta = ctr_by_pos[ctr_by_pos['position'] == k+1]['theta_simple'].values
            simple_val = simple_theta[0] if len(simple_theta) > 0 else np.nan
            raw_ctr = ctr_by_pos[ctr_by_pos['position'] == k+1]['ctr'].values
            raw_ctr_val = raw_ctr[0] * 100 if len(raw_ctr) > 0 else np.nan
            diff = theta_ubm[k] - simple_val if not np.isnan(simple_val) else np.nan
            log(f"  {k+1:<10} {theta_ubm[k]:<15.4f} {simple_val:<12.3f} {diff:<10.3f} {raw_ctr_val:<10.2f}", f)
        log("", f)

        # Monotonicity check
        violations = sum(1 for k in range(len(theta_ubm) - 1) if theta_ubm[k] < theta_ubm[k+1])
        log(f"Monotonicity: {len(theta_ubm)-1-violations}/{len(theta_ubm)-1} positions ({(1-violations/(len(theta_ubm)-1))*100:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Examination After Click (Distance Effect)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: EXAMINATION AFTER CLICK (DISTANCE EFFECT)", f)
        log("-" * 40, f)
        log("", f)

        log("γ_{r,r'}: Examination probability given last click at r'", f)
        log("  UBM's key innovation: models 'restart' behavior after clicking", f)
        log("", f)

        # Show examination probabilities for different last click positions
        log("Examination at position 5 given different last clicks:", f)
        log(f"  Last click at position 0 (none): γ_{5,0} = {ubm.gamma[4, 0]:.4f}", f)
        for r_prime in [1, 2, 3, 4]:
            log(f"  Last click at position {r_prime}: γ_{{5,{r_prime}}} = {ubm.gamma[4, r_prime]:.4f}", f)
        log("", f)

        log("Examination at position 10 given different last clicks:", f)
        log(f"  Last click at position 0 (none): γ_{{10,0}} = {ubm.gamma[9, 0]:.4f}", f)
        for r_prime in [1, 5, 9]:
            log(f"  Last click at position {r_prime}: γ_{{10,{r_prime}}} = {ubm.gamma[9, r_prime]:.4f}", f)
        log("", f)

        # Distance effect analysis
        log("Distance effect: γ_{r, r-d} for position r=10, varying distance d:", f)
        log(f"  {'Distance':<10} {'γ value':<12}", f)
        log(f"  {'-'*10} {'-'*12}", f)
        for d in [1, 2, 3, 5, 7, 9]:
            r_prime = 10 - d
            if r_prime >= 1:
                log(f"  {d:<10} {ubm.gamma[9, r_prime]:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Gamma Matrix Visualization
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: GAMMA MATRIX SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        log("γ_{r,r'} matrix (first 10 positions, r' = 0..5):", f)
        log("  r'=0 is no prior click, r'=1..5 means last click at position 1..5", f)
        log("", f)

        backslash_header = "r\\r'"
        header = f"  {backslash_header:<6}" + "".join([f"{r_prime:<10}" for r_prime in range(6)])
        log(header, f)
        log(f"  {'-'*6}" + "-"*60, f)

        for r in range(10):
            row_str = f"  {r+1:<6}"
            for r_prime in range(6):
                row_str += f"{ubm.gamma[r, r_prime]:<10.4f}"
            log(row_str, f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Prediction Quality
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: PREDICTION QUALITY", f)
        log("-" * 40, f)
        log("", f)

        # Evaluate on session_items
        items_test = items_df[items_df['position'] <= 20].copy()

        # Simple predictions (using no prior click context)
        ubm_pred = items_test.apply(
            lambda row: ubm.predict_ctr(int(row['position']), row['product_id'], last_click=0),
            axis=1
        )

        # Simple baseline
        simple_pred = items_test['position'].map(
            lambda p: ctr_by_pos[ctr_by_pos['position'] == p]['ctr'].values[0]
            if p in ctr_by_pos['position'].values else np.nan
        )

        valid_mask = ~ubm_pred.isna() & ~simple_pred.isna() & (ubm_pred > 0)
        y_true = items_test.loc[valid_mask, 'clicked'].values

        ubm_pred_valid = np.clip(ubm_pred[valid_mask].values, 1e-6, 1-1e-6)
        simple_pred_valid = np.clip(simple_pred[valid_mask].values, 1e-6, 1-1e-6)

        log("Prediction metrics (item-level, no context):", f)
        log(f"  {'Metric':<20} {'Simple':<15} {'UBM':<15}", f)
        log(f"  {'-'*20} {'-'*15} {'-'*15}", f)

        ll_simple = log_loss(y_true, simple_pred_valid)
        ll_ubm = log_loss(y_true, ubm_pred_valid)
        log(f"  {'Log Loss':<20} {ll_simple:<15.4f} {ll_ubm:<15.4f}", f)

        bs_simple = brier_score_loss(y_true, simple_pred_valid)
        bs_ubm = brier_score_loss(y_true, ubm_pred_valid)
        log(f"  {'Brier Score':<20} {bs_simple:<15.4f} {bs_ubm:<15.4f}", f)

        auc_simple = roc_auc_score(y_true, simple_pred_valid)
        auc_ubm = roc_auc_score(y_true, ubm_pred_valid)
        log(f"  {'AUC':<20} {auc_simple:<15.4f} {auc_ubm:<15.4f}", f)

        log("", f)

        ll_improvement = (ll_simple - ll_ubm) / ll_simple * 100
        log(f"Log loss improvement over simple: {ll_improvement:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Session-Level Evaluation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: SESSION-LEVEL EVALUATION", f)
        log("-" * 40, f)
        log("", f)

        log("Evaluating UBM with proper click context...", f)
        log("", f)

        # Compute session-level log-likelihood using proper context
        session_ll_ubm = 0.0
        session_ll_simple = 0.0
        n_evaluated = 0

        for _, row in tqdm(sessions_df.iterrows(), total=len(sessions_df), desc="    Evaluating", disable=False):
            items = row['items'][:20]
            clicks = row['clicks'][:20]

            last_click = 0  # 0 = no prior click

            for k, (item, click) in enumerate(zip(items, clicks)):
                if k >= ubm.max_position:
                    break

                # UBM prediction with context
                if item in ubm.item_to_idx:
                    a_u = ubm.alpha[ubm.item_to_idx[item]]
                else:
                    a_u = np.mean(ubm.alpha)

                gamma_val = ubm.gamma[k, last_click]
                p_ubm = np.clip(gamma_val * a_u, 1e-10, 1 - 1e-10)

                # Simple prediction
                simple_ctr = ctr_by_pos[ctr_by_pos['position'] == k+1]['ctr'].values
                p_simple = np.clip(simple_ctr[0] if len(simple_ctr) > 0 else 0.03, 1e-10, 1 - 1e-10)

                if click == 1:
                    session_ll_ubm += np.log(p_ubm)
                    session_ll_simple += np.log(p_simple)
                    last_click = k + 1  # Update context
                else:
                    session_ll_ubm += np.log(1 - p_ubm)
                    session_ll_simple += np.log(1 - p_simple)

                n_evaluated += 1

        avg_ll_ubm = -session_ll_ubm / n_evaluated
        avg_ll_simple = -session_ll_simple / n_evaluated

        log("Session-level (with click context) metrics:", f)
        log(f"  UBM avg negative LL: {avg_ll_ubm:.4f}", f)
        log(f"  Simple avg negative LL: {avg_ll_simple:.4f}", f)
        log(f"  Improvement: {(avg_ll_simple - avg_ll_ubm) / avg_ll_simple * 100:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Multi-Click Session Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: MULTI-CLICK SESSION ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        multi_click_sessions = sessions_df[sessions_df['n_clicks'] >= 2].copy()

        if len(multi_click_sessions) > 0:
            log(f"N multi-click sessions: {len(multi_click_sessions):,}", f)
            log("", f)

            # Analyze examination patterns after first click
            second_click_exam = []
            for _, row in multi_click_sessions.iterrows():
                clicks = row['clicks']
                click_pos = [i for i, c in enumerate(clicks) if c == 1]
                if len(click_pos) >= 2:
                    first_click = click_pos[0]
                    second_click = click_pos[1]
                    if first_click < ubm.max_position and second_click < ubm.max_position:
                        # Get examination prob for second click position given first
                        exam_prob = ubm.gamma[second_click, first_click + 1]
                        second_click_exam.append({
                            'first_click': first_click + 1,
                            'second_click': second_click + 1,
                            'distance': second_click - first_click,
                            'exam_prob': exam_prob
                        })

            if second_click_exam:
                exam_df = pd.DataFrame(second_click_exam)
                log("Examination probability at second click position:", f)
                log(f"  Mean: {exam_df['exam_prob'].mean():.4f}", f)
                log(f"  By distance from first click:", f)
                for dist, grp in exam_df.groupby('distance'):
                    if len(grp) >= 10:
                        log(f"    Distance {dist}: γ = {grp['exam_prob'].mean():.4f} (n={len(grp)})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Stratified Analysis by Placement
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: STRATIFIED ANALYSIS BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        for placement in sorted(sessions_df['placement'].unique()):
            placement_sessions = sessions_df[sessions_df['placement'] == placement]

            if len(placement_sessions) < 500:
                continue

            log(f"PLACEMENT {placement}:", f)
            log(f"  Sessions: {len(placement_sessions):,}", f)
            log(f"  Multi-click rate: {(placement_sessions['n_clicks'] >= 2).mean()*100:.1f}%", f)

            try:
                ubm_placement = UserBrowsingModel(max_position=10, min_impressions=3)
                ubm_placement.fit_em(placement_sessions, max_iter=30, verbose=False)

                theta_p = ubm_placement.get_position_effects(last_click=0)
                log(f"  γ_{{1,0}}: {theta_p[0]:.3f}", f)
                log(f"  γ_{{5,0}}: {theta_p[4]:.3f}", f)
                log(f"  Mean α: {np.mean(ubm_placement.alpha):.4f}", f)
            except Exception as e:
                log(f"  Could not fit: {str(e)}", f)

            log("", f)

        # -----------------------------------------------------------------
        # Section 10: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 10: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("UBM PARAMETERS:", f)
        log(f"  Mean α (attractiveness): {np.mean(ubm.alpha):.4f}", f)
        log(f"  γ_{{1,0}} (examine position 1, no prior click): {ubm.gamma[0, 0]:.4f}", f)
        log(f"  γ_{{5,0}} (examine position 5, no prior click): {ubm.gamma[4, 0]:.4f}", f)
        log(f"  γ_{{10,0}} (examine position 10, no prior click): {ubm.gamma[9, 0]:.4f}", f)
        log("", f)

        log("POSITION EFFECTS (γ_{r,0}):", f)
        log(f"  Position 1: {ubm.gamma[0, 0]:.4f} (normalized)", f)
        log(f"  Position 5: {ubm.gamma[4, 0]:.4f} ({ubm.gamma[4, 0]/ubm.gamma[0, 0]*100:.1f}% of position 1)", f)
        log(f"  Position 10: {ubm.gamma[9, 0]:.4f} ({ubm.gamma[9, 0]/ubm.gamma[0, 0]*100:.1f}% of position 1)", f)
        if len(ubm.gamma) >= 20:
            log(f"  Position 20: {ubm.gamma[19, 0]:.4f} ({ubm.gamma[19, 0]/ubm.gamma[0, 0]*100:.1f}% of position 1)", f)
        log("", f)

        log("DISTANCE EFFECT (γ_{{10,r'}} for position 10):", f)
        log(f"  No prior click (r'=0): {ubm.gamma[9, 0]:.4f}", f)
        log(f"  Last click at 5 (r'=5): {ubm.gamma[9, 5]:.4f}", f)
        log(f"  Last click at 9 (r'=9): {ubm.gamma[9, 9]:.4f}", f)
        log("", f)

        log("MODEL FIT:", f)
        log(f"  Item-level Log loss: {ll_ubm:.4f}", f)
        log(f"  Item-level AUC: {auc_ubm:.4f}", f)
        log(f"  Session-level avg neg LL: {avg_ll_ubm:.4f}", f)
        log("", f)

        log("KEY FINDINGS:", f)
        log(f"  1. UBM captures distance-to-last-click effect", f)
        log(f"  2. Multi-click rate ({multi_click_rate*100:.1f}%) justifies modeling click context", f)
        if avg_ll_ubm < avg_ll_simple:
            log(f"  3. UBM outperforms simple baseline on session-level LL", f)
        else:
            log(f"  3. Simple baseline competitive on session-level LL", f)
        log("", f)

        log("COMPARISON WITH OTHER MODELS:", f)
        log("  - PBM: position-only examination (no click context)", f)
        log("  - DBN: satisfaction/continuation (partial click context)", f)
        log("  - UBM: full click context via γ_{r,r'} matrix", f)
        log("", f)

        log("=" * 80, f)
        log("UBM ESTIMATION COMPLETE", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
