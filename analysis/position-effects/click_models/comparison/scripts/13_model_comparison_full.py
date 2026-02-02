#!/usr/bin/env python3
"""
Full Click Model Comparison

Compares all implemented click models on same train/test split:
1. PBM (Position-Based Model) - position-only examination
2. DBN (Dynamic Bayesian Network) - satisfaction/continuation
3. SDBN (Simplified DBN) - cascade model with stop-after-click
4. UBM (User Browsing Model) - distance-to-last-click examination
5. Feature Model (Neural) - feature-based attractiveness with position decay

Metrics:
- Log Loss (item-level and session-level)
- AUC
- Brier Score
- Perplexity

Reference: Chuklin et al. (2015) Chapter 7 "Experimental Comparison"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_R2_DIR = BASE_DIR.parent / "data_r2"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "13_model_comparison_full.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# MODEL IMPLEMENTATIONS (simplified versions for comparison)
# =============================================================================

class PositionBasedModel:
    """PBM: P(click) = θ_k × α_i"""

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.theta = None
        self.alpha = None

    def fit(self, sessions_df, verbose=True):
        """Fit via EM algorithm."""
        if verbose:
            print("  Fitting PBM...")

        # Build item index
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        # Preprocess
        sessions_data = []
        for _, row in sessions_df.iterrows():
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]
            item_indices = []
            positions = []
            valid_clicks = []
            for k, (item, click) in enumerate(zip(items, clicks)):
                if item in item_to_idx:
                    item_indices.append(item_to_idx[item])
                    positions.append(k)
                    valid_clicks.append(click)
            if item_indices:
                sessions_data.append({
                    'item_indices': np.array(item_indices),
                    'positions': np.array(positions),
                    'clicks': np.array(valid_clicks)
                })

        # Initialize
        theta = np.ones(self.max_position)
        for k in range(1, self.max_position):
            theta[k] = max(0.1, 1.0 - 0.03 * k)

        alpha = np.full(n_items, 0.03)

        # EM iterations
        for iteration in range(50):
            # M-step accumulators
            theta_num = np.zeros(self.max_position)
            theta_denom = np.zeros(self.max_position)
            alpha_num = np.zeros(n_items)
            alpha_denom = np.zeros(n_items)

            for session in sessions_data:
                for i, (idx, pos, click) in enumerate(zip(
                    session['item_indices'], session['positions'], session['clicks']
                )):
                    if pos < self.max_position:
                        theta_num[pos] += click
                        theta_denom[pos] += alpha[idx]
                        alpha_num[idx] += click
                        alpha_denom[idx] += theta[pos]

            # Update theta
            new_theta = np.ones(self.max_position)
            for k in range(self.max_position):
                if theta_denom[k] > 0:
                    new_theta[k] = theta_num[k] / theta_denom[k]
            if new_theta[0] > 0:
                new_theta = new_theta / new_theta[0]
            new_theta = np.clip(new_theta, 0.001, 1.0)

            # Update alpha
            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                if alpha_denom[i] > 0:
                    new_alpha[i] = alpha_num[i] / alpha_denom[i]
                else:
                    new_alpha[i] = alpha[i]
            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            if np.max(np.abs(new_theta - theta)) < 1e-5 and np.max(np.abs(new_alpha - alpha)) < 1e-5:
                break

            theta = new_theta
            alpha = new_alpha

        self.theta = theta
        self.alpha = alpha
        self.item_to_idx = item_to_idx
        self.n_items = n_items
        return self

    def predict(self, position, product_id):
        if position > self.max_position or position < 1:
            return 0.03
        theta_k = self.theta[position - 1]
        if product_id in self.item_to_idx:
            return theta_k * self.alpha[self.item_to_idx[product_id]]
        return theta_k * np.mean(self.alpha)


class DynamicBayesianNetwork:
    """DBN: P(C|E) = α, P(S|C) = σ, P(E_{k+1}|S=0) = γ"""

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.alpha = None
        self.sigma = 0.5
        self.gamma = 0.8

    def fit(self, sessions_df, verbose=True):
        """Fit via EM algorithm."""
        if verbose:
            print("  Fitting DBN...")

        # Build item index
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        # Preprocess
        sessions_data = []
        for _, row in sessions_df.iterrows():
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]
            item_indices = []
            valid_clicks = []
            for item, click in zip(items, clicks):
                if item in item_to_idx:
                    item_indices.append(item_to_idx[item])
                    valid_clicks.append(click)
            if item_indices:
                sessions_data.append({
                    'item_indices': np.array(item_indices),
                    'clicks': np.array(valid_clicks)
                })

        # Initialize
        alpha = np.full(n_items, 0.03)
        sigma = 0.5
        gamma = 0.8

        for iteration in range(50):
            alpha_num = np.zeros(n_items)
            alpha_denom = np.zeros(n_items)
            sigma_num = 0.0
            sigma_denom = 0.0
            gamma_num = 0.0
            gamma_denom = 0.0

            for session in sessions_data:
                exam_prob = np.ones(len(session['item_indices']))
                for k in range(1, len(session['item_indices'])):
                    if session['clicks'][k-1] == 0:
                        exam_prob[k] = exam_prob[k-1]
                    else:
                        exam_prob[k] = exam_prob[k-1] * (1 - sigma) * gamma

                for k, (idx, click) in enumerate(zip(session['item_indices'], session['clicks'])):
                    alpha_num[idx] += click
                    alpha_denom[idx] += exam_prob[k]

                    if click == 1:
                        sigma_denom += 1
                        if k == len(session['clicks']) - 1:
                            sigma_num += 1
                        elif session['clicks'][k+1:].sum() == 0:
                            sigma_num += sigma
                        else:
                            gamma_num += 1
                            gamma_denom += 1

            # Update
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

            if np.max(np.abs(new_alpha - alpha)) < 1e-5:
                break

            alpha = new_alpha
            sigma = new_sigma
            gamma = new_gamma

        self.alpha = alpha
        self.sigma = sigma
        self.gamma = gamma
        self.item_to_idx = item_to_idx
        self.n_items = n_items
        return self

    def predict(self, position, product_id, n_clicks_before=0):
        if position < 1:
            return 0.03

        avg_alpha = np.mean(self.alpha)
        exam_prob = 1.0
        for k in range(1, position):
            p_no_click = 1 - exam_prob * avg_alpha
            p_click_continue = exam_prob * avg_alpha * (1 - self.sigma) * self.gamma
            exam_prob = p_no_click + p_click_continue

        if product_id in self.item_to_idx:
            return exam_prob * self.alpha[self.item_to_idx[product_id]]
        return exam_prob * avg_alpha


class SimplifiedDBN:
    """SDBN: Click terminates session (cascade model)"""

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.alpha = None
        self.gamma = None

    def fit(self, sessions_df, verbose=True):
        """Fit via MLE."""
        if verbose:
            print("  Fitting SDBN...")

        # Build item index
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        # Count clicks and impressions per item
        item_impressions = np.zeros(n_items)
        item_clicks = np.zeros(n_items)
        position_continues = np.zeros(self.max_position)
        position_total = np.zeros(self.max_position)

        for _, row in sessions_df.iterrows():
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]

            for k, (item, click) in enumerate(zip(items, clicks)):
                if item not in item_to_idx:
                    continue
                idx = item_to_idx[item]
                item_impressions[idx] += 1

                if click == 1:
                    item_clicks[idx] += 1
                    break
                else:
                    if k < self.max_position - 1 and k < len(items) - 1:
                        position_total[k] += 1
                        position_continues[k] += 1

        # Estimate alpha
        alpha = np.zeros(n_items)
        for i in range(n_items):
            if item_impressions[i] > 0:
                alpha[i] = item_clicks[i] / item_impressions[i]
            else:
                alpha[i] = 0.01
        alpha = np.clip(alpha, 0.001, 0.999)

        # Estimate gamma
        gamma = np.ones(self.max_position)
        for k in range(self.max_position):
            if position_total[k] > 0:
                gamma[k] = position_continues[k] / position_total[k]
        gamma = np.clip(gamma, 0.01, 1.0)

        # Compute theta
        theta = np.ones(self.max_position)
        avg_alpha = np.mean(alpha)
        for k in range(1, self.max_position):
            theta[k] = theta[k-1] * (1 - avg_alpha) * gamma[k-1]

        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.item_to_idx = item_to_idx
        self.n_items = n_items
        return self

    def predict(self, position, product_id):
        if position > self.max_position or position < 1:
            return 0.03
        theta_k = self.theta[position - 1]
        if product_id in self.item_to_idx:
            return theta_k * self.alpha[self.item_to_idx[product_id]]
        return theta_k * np.mean(self.alpha)


class UserBrowsingModel:
    """UBM: P(C) = α × γ_{r,r'} with last click context"""

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.alpha = None
        self.gamma = None

    def fit(self, sessions_df, verbose=True):
        """Fit via EM algorithm."""
        if verbose:
            print("  Fitting UBM...")

        # Build item index
        item_counts = {}
        for _, row in sessions_df.iterrows():
            for item in row['items'][:self.max_position]:
                item_counts[item] = item_counts.get(item, 0) + 1

        valid_items = {item for item, count in item_counts.items() if count >= self.min_impressions}
        item_to_idx = {item: idx for idx, item in enumerate(sorted(valid_items))}
        n_items = len(item_to_idx)

        # Preprocess
        sessions_data = []
        for _, row in sessions_df.iterrows():
            items = row['items'][:self.max_position]
            clicks = row['clicks'][:self.max_position]
            item_indices = []
            positions = []
            valid_clicks = []
            for k, (item, click) in enumerate(zip(items, clicks)):
                if item in item_to_idx:
                    item_indices.append(item_to_idx[item])
                    positions.append(k)
                    valid_clicks.append(click)
            if item_indices:
                sessions_data.append({
                    'item_indices': np.array(item_indices),
                    'positions': np.array(positions),
                    'clicks': np.array(valid_clicks)
                })

        # Initialize alpha from CTR
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

        # Initialize gamma
        gamma = np.zeros((self.max_position, self.max_position + 1))
        for r in range(self.max_position):
            gamma[r, 0] = max(0.1, 1.0 - 0.03 * r)
            for r_prime in range(1, self.max_position + 1):
                if r_prime - 1 < r:
                    distance = r - (r_prime - 1)
                    gamma[r, r_prime] = max(0.1, 1.0 - 0.05 * distance)
                else:
                    gamma[r, r_prime] = 0.1

        # EM iterations
        for iteration in range(50):
            alpha_num = np.zeros(n_items)
            alpha_denom = np.zeros(n_items)
            gamma_num = np.zeros((self.max_position, self.max_position + 1))
            gamma_denom = np.zeros((self.max_position, self.max_position + 1))

            for session in sessions_data:
                last_click = 0
                for k in range(len(session['item_indices'])):
                    pos = session['positions'][k]
                    if pos >= self.max_position:
                        continue
                    idx = session['item_indices'][k]
                    click = session['clicks'][k]

                    exam_prob = gamma[pos, last_click]
                    a_u = alpha[idx]
                    p_click = np.clip(exam_prob * a_u, 1e-10, 1 - 1e-10)

                    if click == 1:
                        exam_given_obs = 1.0
                    else:
                        p_exam_no_click = exam_prob * (1 - a_u)
                        p_not_click = 1 - p_click
                        exam_given_obs = p_exam_no_click / p_not_click if p_not_click > 1e-10 else exam_prob

                    alpha_num[idx] += click
                    alpha_denom[idx] += exam_given_obs
                    gamma_num[pos, last_click] += exam_given_obs
                    gamma_denom[pos, last_click] += 1

                    if click == 1:
                        last_click = pos + 1

            # Update
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

            if np.max(np.abs(new_alpha - alpha)) < 1e-5 and np.max(np.abs(new_gamma - gamma)) < 1e-5:
                break

            alpha = new_alpha
            gamma = new_gamma

        self.alpha = alpha
        self.gamma = gamma
        self.item_to_idx = item_to_idx
        self.n_items = n_items
        return self

    def predict(self, position, product_id, last_click=0):
        if position > self.max_position or position < 1:
            return 0.03
        gamma_val = self.gamma[position - 1, last_click]
        if product_id in self.item_to_idx:
            return gamma_val * self.alpha[self.item_to_idx[product_id]]
        return gamma_val * np.mean(self.alpha)


class SimpleBaseline:
    """Simple position-based CTR baseline."""

    def __init__(self, max_position=20):
        self.max_position = max_position
        self.ctr_by_pos = None

    def fit(self, sessions_df, verbose=True):
        if verbose:
            print("  Fitting Simple Baseline...")

        # Compute CTR by position
        pos_clicks = np.zeros(self.max_position)
        pos_impr = np.zeros(self.max_position)

        for _, row in sessions_df.iterrows():
            clicks = row['clicks'][:self.max_position]
            for k, click in enumerate(clicks):
                if k < self.max_position:
                    pos_impr[k] += 1
                    pos_clicks[k] += click

        self.ctr_by_pos = np.zeros(self.max_position)
        for k in range(self.max_position):
            if pos_impr[k] > 0:
                self.ctr_by_pos[k] = pos_clicks[k] / pos_impr[k]
            else:
                self.ctr_by_pos[k] = 0.03

        return self

    def predict(self, position, product_id=None):
        if position > self.max_position or position < 1:
            return 0.03
        return self.ctr_by_pos[position - 1]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_item_level(model, sessions_df, model_name, use_context=False):
    """Evaluate model at item level."""
    y_true = []
    y_pred = []

    for _, row in sessions_df.iterrows():
        items = row['items'][:20]
        clicks = row['clicks'][:20]
        last_click = 0

        for k, (item, click) in enumerate(zip(items, clicks)):
            position = k + 1

            if use_context and hasattr(model, 'gamma') and model.gamma is not None:
                pred = model.predict(position, item, last_click)
            else:
                pred = model.predict(position, item)

            y_true.append(click)
            y_pred.append(pred)

            if click == 1:
                last_click = position

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-6, 1 - 1e-6)

    return {
        'model': model_name,
        'log_loss': log_loss(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'perplexity': np.exp(log_loss(y_true, y_pred)),
        'n_samples': len(y_true)
    }


def evaluate_session_level(model, sessions_df, model_name, use_context=False):
    """Evaluate model at session level (average log-likelihood per session)."""
    session_lls = []

    for _, row in sessions_df.iterrows():
        items = row['items'][:20]
        clicks = row['clicks'][:20]
        last_click = 0
        session_ll = 0.0

        for k, (item, click) in enumerate(zip(items, clicks)):
            position = k + 1

            if use_context and hasattr(model, 'gamma') and model.gamma is not None:
                pred = model.predict(position, item, last_click)
            else:
                pred = model.predict(position, item)

            pred = np.clip(pred, 1e-10, 1 - 1e-10)

            if click == 1:
                session_ll += np.log(pred)
                last_click = position
            else:
                session_ll += np.log(1 - pred)

        session_lls.append(session_ll)

    avg_ll = np.mean(session_lls)

    return {
        'model': model_name,
        'avg_session_ll': avg_ll,
        'avg_neg_ll': -avg_ll / np.mean([len(row['items'][:20]) for _, row in sessions_df.iterrows()]),
        'session_perplexity': np.exp(-avg_ll / np.mean([len(row['items'][:20]) for _, row in sessions_df.iterrows()])),
        'n_sessions': len(session_lls)
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("FULL CLICK MODEL COMPARISON", f)
        log("=" * 80, f)
        log("", f)

        log("MODELS COMPARED:", f)
        log("  1. Simple Baseline: Position-specific CTR", f)
        log("  2. PBM: Position-Based Model (θ_k × α_i)", f)
        log("  3. DBN: Dynamic Bayesian Network (satisfaction/continuation)", f)
        log("  4. SDBN: Simplified DBN (cascade, stop-after-click)", f)
        log("  5. UBM: User Browsing Model (γ_{r,r'} distance effect)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Load Data
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        sessions_df = pd.read_parquet(DATA_DIR / "sessions.parquet")
        items_df = pd.read_parquet(DATA_DIR / "session_items.parquet")

        log(f"Loaded sessions: {len(sessions_df):,}", f)
        log(f"Loaded items: {len(items_df):,}", f)
        log("", f)

        # Data statistics
        total_clicks = items_df['clicked'].sum()
        total_impressions = len(items_df)
        overall_ctr = total_clicks / total_impressions

        log("Data statistics:", f)
        log(f"  Total impressions: {total_impressions:,}", f)
        log(f"  Total clicks: {int(total_clicks):,}", f)
        log(f"  Overall CTR: {overall_ctr*100:.2f}%", f)
        log(f"  Multi-click rate: {(sessions_df['n_clicks'] >= 2).mean()*100:.1f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Train/Test Split
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: TRAIN/TEST SPLIT", f)
        log("-" * 40, f)
        log("", f)

        # Split by auction_id for proper evaluation
        unique_auctions = sessions_df['auction_id'].unique()
        train_auctions, test_auctions = train_test_split(
            unique_auctions, test_size=0.2, random_state=42
        )

        train_sessions = sessions_df[sessions_df['auction_id'].isin(train_auctions)]
        test_sessions = sessions_df[sessions_df['auction_id'].isin(test_auctions)]

        log(f"Train sessions: {len(train_sessions):,}", f)
        log(f"Test sessions: {len(test_sessions):,}", f)
        log("", f)

        # Click rates
        train_clicks = sum(sum(row['clicks'][:20]) for _, row in train_sessions.iterrows())
        train_impr = sum(len(row['clicks'][:20]) for _, row in train_sessions.iterrows())
        test_clicks = sum(sum(row['clicks'][:20]) for _, row in test_sessions.iterrows())
        test_impr = sum(len(row['clicks'][:20]) for _, row in test_sessions.iterrows())

        log(f"Train CTR: {train_clicks/train_impr*100:.2f}%", f)
        log(f"Test CTR: {test_clicks/test_impr*100:.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Fit Models
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: MODEL FITTING", f)
        log("-" * 40, f)
        log("", f)

        models = {}

        # Simple Baseline
        models['Simple'] = SimpleBaseline(max_position=20)
        models['Simple'].fit(train_sessions, verbose=True)

        # PBM
        models['PBM'] = PositionBasedModel(max_position=20, min_impressions=5)
        models['PBM'].fit(train_sessions, verbose=True)

        # DBN
        models['DBN'] = DynamicBayesianNetwork(max_position=20, min_impressions=5)
        models['DBN'].fit(train_sessions, verbose=True)

        # SDBN
        models['SDBN'] = SimplifiedDBN(max_position=20, min_impressions=5)
        models['SDBN'].fit(train_sessions, verbose=True)

        # UBM
        models['UBM'] = UserBrowsingModel(max_position=20, min_impressions=5)
        models['UBM'].fit(train_sessions, verbose=True)

        log("", f)
        log("All models fitted.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Item-Level Evaluation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: ITEM-LEVEL EVALUATION (TEST SET)", f)
        log("-" * 40, f)
        log("", f)

        log("Evaluating models at item level...", f)
        log("", f)

        item_results = []
        for name, model in tqdm(models.items(), desc="  Evaluating"):
            use_context = (name == 'UBM')
            result = evaluate_item_level(model, test_sessions, name, use_context=use_context)
            item_results.append(result)

        # Sort by log loss
        item_results = sorted(item_results, key=lambda x: x['log_loss'])

        log("Item-level metrics (sorted by Log Loss):", f)
        log(f"  {'Model':<10} {'Log Loss':<12} {'AUC':<10} {'Brier':<10} {'Perplexity':<12}", f)
        log(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", f)

        for r in item_results:
            log(f"  {r['model']:<10} {r['log_loss']:<12.4f} {r['auc']:<10.4f} {r['brier']:<10.4f} {r['perplexity']:<12.4f}", f)
        log("", f)

        # Best model
        best_item = item_results[0]
        log(f"Best item-level model: {best_item['model']} (Log Loss = {best_item['log_loss']:.4f})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 5: Session-Level Evaluation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: SESSION-LEVEL EVALUATION (TEST SET)", f)
        log("-" * 40, f)
        log("", f)

        log("Evaluating models at session level...", f)
        log("", f)

        session_results = []
        for name, model in tqdm(models.items(), desc="  Evaluating"):
            use_context = (name == 'UBM')
            result = evaluate_session_level(model, test_sessions, name, use_context=use_context)
            session_results.append(result)

        # Sort by avg negative LL (lower is better)
        session_results = sorted(session_results, key=lambda x: x['avg_neg_ll'])

        log("Session-level metrics (sorted by Avg Neg LL):", f)
        log(f"  {'Model':<10} {'Avg Session LL':<15} {'Avg Neg LL':<12} {'Perplexity':<12}", f)
        log(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*12}", f)

        for r in session_results:
            log(f"  {r['model']:<10} {r['avg_session_ll']:<15.2f} {r['avg_neg_ll']:<12.4f} {r['session_perplexity']:<12.4f}", f)
        log("", f)

        # Best model
        best_session = session_results[0]
        log(f"Best session-level model: {best_session['model']} (Avg Neg LL = {best_session['avg_neg_ll']:.4f})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Model Parameters Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: MODEL PARAMETERS SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        # PBM parameters
        log("PBM Position Effects (θ_k):", f)
        for k in [1, 5, 10, 15, 20]:
            if k <= len(models['PBM'].theta):
                log(f"  θ_{k} = {models['PBM'].theta[k-1]:.4f}", f)
        log(f"  Mean α = {np.mean(models['PBM'].alpha):.4f}", f)
        log("", f)

        # DBN parameters
        log("DBN Parameters:", f)
        log(f"  σ (satisfaction) = {models['DBN'].sigma:.4f}", f)
        log(f"  γ (continuation) = {models['DBN'].gamma:.4f}", f)
        log(f"  Mean α = {np.mean(models['DBN'].alpha):.4f}", f)
        log("", f)

        # SDBN parameters
        log("SDBN Parameters:", f)
        log(f"  Mean γ (continuation) = {np.mean(models['SDBN'].gamma[:10]):.4f}", f)
        log(f"  Mean α = {np.mean(models['SDBN'].alpha):.4f}", f)
        log("", f)

        # UBM parameters
        log("UBM Parameters:", f)
        log(f"  γ_{{1,0}} = {models['UBM'].gamma[0, 0]:.4f}", f)
        log(f"  γ_{{5,0}} = {models['UBM'].gamma[4, 0]:.4f}", f)
        log(f"  γ_{{10,0}} = {models['UBM'].gamma[9, 0]:.4f}", f)
        log(f"  Mean α = {np.mean(models['UBM'].alpha):.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 7: Statistical Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: RELATIVE PERFORMANCE", f)
        log("-" * 40, f)
        log("", f)

        simple_ll = [r for r in item_results if r['model'] == 'Simple'][0]['log_loss']

        log("Log Loss improvement over Simple Baseline:", f)
        for r in item_results:
            if r['model'] != 'Simple':
                improvement = (simple_ll - r['log_loss']) / simple_ll * 100
                log(f"  {r['model']}: {improvement:+.2f}%", f)
        log("", f)

        # AUC comparison
        log("AUC improvement over Simple Baseline:", f)
        simple_auc = [r for r in item_results if r['model'] == 'Simple'][0]['auc']
        for r in item_results:
            if r['model'] != 'Simple':
                improvement = (r['auc'] - simple_auc) / simple_auc * 100
                log(f"  {r['model']}: {improvement:+.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Model Ranking
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: FINAL MODEL RANKING", f)
        log("-" * 40, f)
        log("", f)

        # Combined ranking
        model_ranks = {}
        for i, r in enumerate(item_results):
            model_ranks[r['model']] = {'item_ll_rank': i + 1}

        for i, r in enumerate(session_results):
            if r['model'] in model_ranks:
                model_ranks[r['model']]['session_ll_rank'] = i + 1

        item_results_by_auc = sorted(item_results, key=lambda x: -x['auc'])
        for i, r in enumerate(item_results_by_auc):
            model_ranks[r['model']]['auc_rank'] = i + 1

        log("Model rankings:", f)
        log(f"  {'Model':<10} {'LL Rank':<10} {'AUC Rank':<10} {'Session Rank':<12} {'Avg Rank':<10}", f)
        log(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}", f)

        for model, ranks in model_ranks.items():
            ll_rank = ranks.get('item_ll_rank', 0)
            auc_rank = ranks.get('auc_rank', 0)
            sess_rank = ranks.get('session_ll_rank', 0)
            avg_rank = (ll_rank + auc_rank + sess_rank) / 3
            log(f"  {model:<10} {ll_rank:<10} {auc_rank:<10} {sess_rank:<12} {avg_rank:<10.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 9: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 9: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("KEY FINDINGS:", f)
        log("", f)

        log(f"1. Best item-level Log Loss: {best_item['model']} ({best_item['log_loss']:.4f})", f)
        log(f"2. Best session-level LL: {best_session['model']} ({best_session['avg_neg_ll']:.4f})", f)
        log(f"3. Best AUC: {item_results_by_auc[0]['model']} ({item_results_by_auc[0]['auc']:.4f})", f)
        log("", f)

        log("MODEL CHARACTERISTICS:", f)
        log("  - Simple: Position-only, no item effects", f)
        log("  - PBM: Separates position (θ) and item (α) effects", f)
        log("  - DBN: Models satisfaction/continuation dynamics", f)
        log("  - SDBN: Cascade model, assumes single click", f)
        log("  - UBM: Models examination given any prior click position", f)
        log("", f)

        log("RECOMMENDATIONS:", f)
        if best_item['model'] == 'UBM':
            log("  - UBM best for click prediction (captures click context)", f)
        elif best_item['model'] == 'PBM':
            log("  - PBM best for click prediction (simpler, robust)", f)
        else:
            log(f"  - {best_item['model']} best for click prediction", f)

        log("  - Use PBM for position effect estimation (interpretable θ_k)", f)
        log("  - Use DBN for LTR training (separates attractiveness/satisfaction)", f)
        log("", f)

        log("=" * 80, f)
        log("MODEL COMPARISON COMPLETE", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
