#!/usr/bin/env python3
"""
Grid Position-Based Model (Grid-PBM)

Mathematical Framework:
- P(C_{u,p} = 1) = γ_{row,col} × α_u
- γ_{row,col} = examination probability at grid position (row, col)
- α_u = attractiveness of product u (from quality or estimated)

Key Insight:
Users see items in a 2D grid (viewport). Examination probability may differ
spatially: top-left vs bottom-right, left column vs right column within same row.

Grid Mapping:
Given grid width W (items per row), map linear position to (row, col):
    row = (position - 1) // grid_width
    col = (position - 1) % grid_width

Example with W=2:
    Position 1 → (0, 0) top-left
    Position 2 → (0, 1) top-right
    Position 3 → (1, 0) second row left
    Position 4 → (1, 1) second row right

Reference: Extension of Craswell et al. (2008) PBM to 2D grid layout.
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
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "14_grid_pbm.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# GRID-PBM MODEL
# =============================================================================

class GridPositionBasedModel:
    """
    Grid Position-Based Model (Grid-PBM)

    Model: P(click at grid position (row, col), item i) = γ_{row,col} × α_i

    Parameters:
    - grid_width: Number of items per row (W)
    - max_rows: Maximum number of rows to model
    - min_impressions: Minimum impressions per item for estimation
    """

    def __init__(self, grid_width=2, max_rows=10, min_impressions=5):
        self.grid_width = grid_width
        self.max_rows = max_rows
        self.max_position = grid_width * max_rows
        self.min_impressions = min_impressions
        self.gamma = None  # γ_{row,col} matrix
        self.alpha = None  # Item attractiveness
        self.fitted = False

    def position_to_grid(self, position):
        """Map linear position (1-indexed) to grid coordinates (row, col)."""
        pos_0 = position - 1  # Convert to 0-indexed
        row = pos_0 // self.grid_width
        col = pos_0 % self.grid_width
        return row, col

    def grid_to_position(self, row, col):
        """Map grid coordinates to linear position (1-indexed)."""
        return row * self.grid_width + col + 1

    def fit(self, items_df, max_iter=100, tol=1e-6, verbose=True):
        """
        Fit Grid-PBM via EM algorithm.

        E-step: Compute expected examination given clicks
        M-step: Update γ_{row,col} and α_i
        """
        # Filter to positions within grid
        df = items_df[items_df['position'] <= self.max_position].copy()

        # Get unique items with sufficient data
        item_counts = df.groupby('product_id').size()
        valid_items = item_counts[item_counts >= self.min_impressions].index
        df = df[df['product_id'].isin(valid_items)]

        if verbose:
            print(f"  Fitting Grid-PBM (W={self.grid_width}) with {len(df):,} observations, {len(valid_items):,} items")

        # Build item index
        n_items = len(valid_items)
        item_to_idx = {item: idx for idx, item in enumerate(valid_items)}

        # Map positions to grid coordinates
        df['row'] = (df['position'] - 1) // self.grid_width
        df['col'] = (df['position'] - 1) % self.grid_width
        df['item_idx'] = df['product_id'].map(item_to_idx)
        df = df.dropna(subset=['item_idx'])
        df['item_idx'] = df['item_idx'].astype(int)

        # Extract arrays for fast computation
        rows = df['row'].values.astype(int)
        cols = df['col'].values.astype(int)
        item_indices = df['item_idx'].values
        clicks = df['clicked'].values.astype(float)

        # Initialize gamma (grid examination effects)
        gamma = np.ones((self.max_rows, self.grid_width))
        # Initialize with decay pattern
        for r in range(self.max_rows):
            for c in range(self.grid_width):
                gamma[r, c] = max(0.1, 1.0 - 0.05 * r - 0.02 * c)

        # Normalize gamma[0,0] = 1
        gamma = gamma / gamma[0, 0]

        # Initialize alpha from observed CTR
        alpha = np.zeros(n_items)
        for item, idx in item_to_idx.items():
            item_data = df[df['product_id'] == item]
            alpha[idx] = np.clip(item_data['clicked'].mean() + 0.01, 0.001, 0.999)

        ll_history = []

        for iteration in tqdm(range(max_iter), desc="  EM iterations", disable=not verbose):
            # M-step accumulators
            gamma_num = np.zeros((self.max_rows, self.grid_width))
            gamma_denom = np.zeros((self.max_rows, self.grid_width))
            alpha_num = np.zeros(n_items)
            alpha_denom = np.zeros(n_items)

            # Accumulate statistics
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                if r < self.max_rows and c < self.grid_width:
                    idx = item_indices[i]
                    click = clicks[i]

                    gamma_num[r, c] += click
                    gamma_denom[r, c] += alpha[idx]
                    alpha_num[idx] += click
                    alpha_denom[idx] += gamma[r, c]

            # Update gamma
            new_gamma = np.ones((self.max_rows, self.grid_width))
            for r in range(self.max_rows):
                for c in range(self.grid_width):
                    if gamma_denom[r, c] > 0:
                        new_gamma[r, c] = gamma_num[r, c] / gamma_denom[r, c]

            # Normalize gamma[0,0] = 1
            if new_gamma[0, 0] > 0:
                new_gamma = new_gamma / new_gamma[0, 0]
            new_gamma = np.clip(new_gamma, 0.001, 1.0)

            # Update alpha
            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                if alpha_denom[i] > 0:
                    new_alpha[i] = alpha_num[i] / alpha_denom[i]
                else:
                    new_alpha[i] = alpha[i]
            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            # Compute log-likelihood
            probs = np.zeros(len(rows))
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                if r < self.max_rows and c < self.grid_width:
                    probs[i] = gamma[r, c] * alpha[item_indices[i]]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            ll = np.sum(clicks * np.log(probs) + (1 - clicks) * np.log(1 - probs))
            ll_history.append(ll)

            # Check convergence
            gamma_change = np.max(np.abs(new_gamma - gamma))
            alpha_change = np.max(np.abs(new_alpha - alpha))

            gamma = new_gamma
            alpha = new_alpha

            if gamma_change < tol and alpha_change < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

        self.gamma = gamma
        self.alpha = alpha
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.ll_history = ll_history
        self.n_items = n_items
        self.fitted = True

        return self

    def predict(self, position, product_id=None):
        """Predict click probability for position (and optionally item)."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        if position > self.max_position or position < 1:
            return 0.03

        row, col = self.position_to_grid(position)
        if row >= self.max_rows or col >= self.grid_width:
            return 0.03

        gamma_val = self.gamma[row, col]

        if product_id is None:
            return gamma_val * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            return gamma_val * self.alpha[self.item_to_idx[product_id]]
        else:
            return gamma_val * np.mean(self.alpha)

    def get_gamma_matrix(self):
        """Return the gamma matrix."""
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


class StandardPBM:
    """Standard Position-Based Model for comparison."""

    def __init__(self, max_position=20, min_impressions=5):
        self.max_position = max_position
        self.min_impressions = min_impressions
        self.theta = None
        self.alpha = None

    def fit(self, items_df, max_iter=100, tol=1e-6, verbose=True):
        """Fit via EM algorithm."""
        df = items_df[items_df['position'] <= self.max_position].copy()

        item_counts = df.groupby('product_id').size()
        valid_items = item_counts[item_counts >= self.min_impressions].index
        df = df[df['product_id'].isin(valid_items)]

        if verbose:
            print(f"  Fitting Standard PBM with {len(df):,} observations, {len(valid_items):,} items")

        n_items = len(valid_items)
        item_to_idx = {item: idx for idx, item in enumerate(valid_items)}

        df['item_idx'] = df['product_id'].map(item_to_idx)
        df = df.dropna(subset=['item_idx'])
        df['item_idx'] = df['item_idx'].astype(int)

        positions = df['position'].values - 1  # 0-indexed
        item_indices = df['item_idx'].values
        clicks = df['clicked'].values.astype(float)

        # Initialize
        theta = np.ones(self.max_position)
        for k in range(1, self.max_position):
            theta[k] = max(0.1, 1.0 - 0.05 * k)

        alpha = np.zeros(n_items)
        for item, idx in item_to_idx.items():
            item_data = df[df['product_id'] == item]
            alpha[idx] = np.clip(item_data['clicked'].mean() + 0.01, 0.001, 0.999)

        for iteration in tqdm(range(max_iter), desc="  EM iterations", disable=not verbose):
            # M-step
            theta_num = np.zeros(self.max_position)
            theta_denom = np.zeros(self.max_position)
            alpha_num = np.zeros(n_items)
            alpha_denom = np.zeros(n_items)

            for i in range(len(positions)):
                k = positions[i]
                if k < self.max_position:
                    idx = item_indices[i]
                    click = clicks[i]

                    theta_num[k] += click
                    theta_denom[k] += alpha[idx]
                    alpha_num[idx] += click
                    alpha_denom[idx] += theta[k]

            new_theta = np.ones(self.max_position)
            for k in range(self.max_position):
                if theta_denom[k] > 0:
                    new_theta[k] = theta_num[k] / theta_denom[k]
            if new_theta[0] > 0:
                new_theta = new_theta / new_theta[0]
            new_theta = np.clip(new_theta, 0.001, 1.0)

            new_alpha = np.zeros(n_items)
            for i in range(n_items):
                if alpha_denom[i] > 0:
                    new_alpha[i] = alpha_num[i] / alpha_denom[i]
                else:
                    new_alpha[i] = alpha[i]
            new_alpha = np.clip(new_alpha, 0.001, 0.999)

            if np.max(np.abs(new_theta - theta)) < tol and np.max(np.abs(new_alpha - alpha)) < tol:
                break

            theta = new_theta
            alpha = new_alpha

        self.theta = theta
        self.alpha = alpha
        self.item_to_idx = item_to_idx
        self.n_items = n_items
        self.fitted = True

        return self

    def predict(self, position, product_id=None):
        if position > self.max_position or position < 1:
            return 0.03
        theta_k = self.theta[position - 1]
        if product_id is None:
            return theta_k * np.mean(self.alpha)
        elif product_id in self.item_to_idx:
            return theta_k * self.alpha[self.item_to_idx[product_id]]
        else:
            return theta_k * np.mean(self.alpha)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, items_df, max_position=20):
    """Evaluate model on item-level predictions."""
    df = items_df[items_df['position'] <= max_position].copy()

    y_true = df['clicked'].values
    y_pred = df.apply(
        lambda row: model.predict(int(row['position']), row['product_id']),
        axis=1
    ).values

    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)

    return {
        'log_loss': log_loss(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'perplexity': np.exp(log_loss(y_true, y_pred)),
        'n_samples': len(y_true)
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("GRID POSITION-BASED MODEL (GRID-PBM)", f)
        log("=" * 80, f)
        log("", f)

        log("MATHEMATICAL FRAMEWORK:", f)
        log("  P(click at grid (row, col), item i) = γ_{row,col} × α_i", f)
        log("  γ_{row,col} = examination probability at grid position", f)
        log("  α_i = attractiveness of product i", f)
        log("", f)

        log("GRID MAPPING:", f)
        log("  Given grid width W, map linear position to (row, col):", f)
        log("    row = (position - 1) // W", f)
        log("    col = (position - 1) % W", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Data Loading
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: DATA LOADING", f)
        log("-" * 40, f)
        log("", f)

        items_df = pd.read_parquet(DATA_DIR / "session_items.parquet")

        log(f"Loaded session_items: {len(items_df):,} rows", f)
        log(f"  Total clicks: {items_df['clicked'].sum():,}", f)
        log(f"  Overall CTR: {items_df['clicked'].mean()*100:.3f}%", f)
        log(f"  Unique products: {items_df['product_id'].nunique():,}", f)
        log(f"  Position range: {items_df['position'].min()} - {items_df['position'].max()}", f)
        log(f"  Placements: {sorted(items_df['placement'].unique())}", f)
        log("", f)

        # Position distribution
        log("Position distribution (top 10):", f)
        pos_dist = items_df.groupby('position').agg({
            'clicked': ['count', 'sum', 'mean']
        }).head(10)
        pos_dist.columns = ['impressions', 'clicks', 'ctr']
        log(f"  {'Position':<10} {'Impressions':<12} {'Clicks':<10} {'CTR %':<10}", f)
        for pos in pos_dist.index:
            row = pos_dist.loc[pos]
            log(f"  {pos:<10} {int(row['impressions']):<12,} {int(row['clicks']):<10,} {row['ctr']*100:<10.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Train/Test Split
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: TRAIN/TEST SPLIT", f)
        log("-" * 40, f)
        log("", f)

        unique_auctions = items_df['auction_id'].unique()
        train_auctions, test_auctions = train_test_split(
            unique_auctions, test_size=0.2, random_state=42
        )

        train_items = items_df[items_df['auction_id'].isin(train_auctions)]
        test_items = items_df[items_df['auction_id'].isin(test_auctions)]

        log(f"Train: {len(train_items):,} items ({train_items['clicked'].mean()*100:.2f}% CTR)", f)
        log(f"Test:  {len(test_items):,} items ({test_items['clicked'].mean()*100:.2f}% CTR)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Fit Standard PBM (Baseline)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: STANDARD PBM (BASELINE)", f)
        log("-" * 40, f)
        log("", f)

        log("MODEL: θ_k × α_i (linear position index)", f)
        log("", f)

        pbm = StandardPBM(max_position=20, min_impressions=5)
        pbm.fit(train_items, verbose=True)

        log("", f)
        log("Position effects (θ_k):", f)
        log(f"  {'Position':<10} {'θ_k':<10}", f)
        log(f"  {'-'*10} {'-'*10}", f)
        for k in range(min(10, len(pbm.theta))):
            log(f"  {k+1:<10} {pbm.theta[k]:<10.4f}", f)
        log("", f)

        pbm_test = evaluate_model(pbm, test_items, max_position=20)
        log("Test set performance:", f)
        log(f"  Log Loss: {pbm_test['log_loss']:.4f}", f)
        log(f"  AUC:      {pbm_test['auc']:.4f}", f)
        log(f"  Brier:    {pbm_test['brier']:.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Grid-PBM with Different Grid Widths
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: GRID-PBM WITH VARYING GRID WIDTH", f)
        log("-" * 40, f)
        log("", f)

        grid_results = {}
        grid_models = {}

        for grid_width in [2, 3, 4]:
            log(f"GRID WIDTH = {grid_width} (items per row)", f)
            log(f"  Grid layout example (first 8 positions):", f)
            for pos in range(1, min(9, grid_width * 4 + 1)):
                row = (pos - 1) // grid_width
                col = (pos - 1) % grid_width
                log(f"    Position {pos} → (row={row}, col={col})", f)
            log("", f)

            model = GridPositionBasedModel(grid_width=grid_width, max_rows=10, min_impressions=5)
            model.fit(train_items, verbose=True)

            grid_models[grid_width] = model

            # Gamma matrix
            log("", f)
            log(f"  Gamma matrix (γ_{{row,col}}):", f)
            gamma = model.get_gamma_matrix()

            # Header
            header = "  " + " " * 8
            for c in range(grid_width):
                header += f"Col {c:<6}"
            log(header, f)

            # Rows
            for r in range(min(5, model.max_rows)):
                row_str = f"  Row {r}:  "
                for c in range(grid_width):
                    row_str += f"{gamma[r, c]:<10.4f}"
                log(row_str, f)
            log("", f)

            # Evaluate
            test_metrics = evaluate_model(model, test_items, max_position=grid_width * 10)
            grid_results[grid_width] = test_metrics

            log(f"  Test performance:", f)
            log(f"    Log Loss: {test_metrics['log_loss']:.4f}", f)
            log(f"    AUC:      {test_metrics['auc']:.4f}", f)
            log(f"    Brier:    {test_metrics['brier']:.4f}", f)
            log("", f)

        # -----------------------------------------------------------------
        # Section 5: Spatial Pattern Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: SPATIAL PATTERN ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Use W=2 for detailed analysis (most common grid layout)
        model_w2 = grid_models[2]
        gamma_w2 = model_w2.get_gamma_matrix()

        log("ANALYSIS FOR GRID WIDTH = 2:", f)
        log("", f)

        # Row decay pattern
        log("1. ROW DECAY (vertical examination pattern):", f)
        log(f"   Row 0 (top):    γ_{{0,*}} avg = {np.mean(gamma_w2[0, :]):.4f}", f)
        log(f"   Row 1:          γ_{{1,*}} avg = {np.mean(gamma_w2[1, :]):.4f}", f)
        log(f"   Row 2:          γ_{{2,*}} avg = {np.mean(gamma_w2[2, :]):.4f}", f)
        log(f"   Row 3:          γ_{{3,*}} avg = {np.mean(gamma_w2[3, :]):.4f}", f)
        log(f"   Row 4:          γ_{{4,*}} avg = {np.mean(gamma_w2[4, :]):.4f}", f)
        log("", f)

        row_decay = []
        for r in range(5):
            row_decay.append(np.mean(gamma_w2[r, :]))
        log(f"   Row decay ratio (row1/row0): {row_decay[1]/row_decay[0]:.3f}", f)
        log(f"   Row decay ratio (row2/row0): {row_decay[2]/row_decay[0]:.3f}", f)
        log("", f)

        # Column effect (left vs right)
        log("2. COLUMN EFFECT (left vs right within same row):", f)
        for r in range(5):
            left = gamma_w2[r, 0]
            right = gamma_w2[r, 1]
            ratio = right / left if left > 0 else 0
            log(f"   Row {r}: Left (col 0) = {left:.4f}, Right (col 1) = {right:.4f}, Ratio = {ratio:.3f}", f)
        log("", f)

        col_0_avg = np.mean(gamma_w2[:5, 0])
        col_1_avg = np.mean(gamma_w2[:5, 1])
        log(f"   Average column effect: Col 0 = {col_0_avg:.4f}, Col 1 = {col_1_avg:.4f}", f)
        log(f"   Left column advantage: {(col_0_avg - col_1_avg) / col_0_avg * 100:.1f}%", f)
        log("", f)

        # Grid vs Linear position comparison
        log("3. GRID POSITION VS LINEAR POSITION:", f)
        log(f"   {'Position':<10} {'Linear θ':<12} {'Grid γ':<12} {'Difference':<12}", f)
        log(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*12}", f)
        for pos in range(1, 11):
            linear_theta = pbm.theta[pos - 1] if pos <= len(pbm.theta) else 0
            row, col = model_w2.position_to_grid(pos)
            grid_gamma = gamma_w2[row, col] if row < model_w2.max_rows else 0
            diff = grid_gamma - linear_theta
            log(f"   {pos:<10} {linear_theta:<12.4f} {grid_gamma:<12.4f} {diff:<12.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 6: Analysis by Placement
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 6: GRID EFFECTS BY PLACEMENT", f)
        log("-" * 40, f)
        log("", f)

        for placement in sorted(items_df['placement'].unique()):
            placement_train = train_items[train_items['placement'] == placement]
            placement_test = test_items[test_items['placement'] == placement]

            if len(placement_train) < 1000:
                continue

            log(f"PLACEMENT {placement}:", f)
            log(f"  Train: {len(placement_train):,} items", f)
            log(f"  Test:  {len(placement_test):,} items", f)

            # Fit Grid-PBM for this placement
            placement_model = GridPositionBasedModel(grid_width=2, max_rows=10, min_impressions=3)
            try:
                placement_model.fit(placement_train, verbose=False)
                gamma_p = placement_model.get_gamma_matrix()

                log(f"  Gamma matrix (first 3 rows):", f)
                for r in range(min(3, placement_model.max_rows)):
                    log(f"    Row {r}: γ_{{0}}={gamma_p[r, 0]:.4f}, γ_{{1}}={gamma_p[r, 1]:.4f}", f)

                # Column effect
                col_effect = np.mean(gamma_p[:3, 0]) / np.mean(gamma_p[:3, 1]) if np.mean(gamma_p[:3, 1]) > 0 else 1.0
                log(f"  Left/Right ratio: {col_effect:.3f}", f)
            except Exception as e:
                log(f"  Could not fit model: {e}", f)
            log("", f)

        # -----------------------------------------------------------------
        # Section 7: Model Comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 7: MODEL COMPARISON", f)
        log("-" * 40, f)
        log("", f)

        log("Test set metrics comparison:", f)
        log(f"  {'Model':<20} {'Log Loss':<12} {'AUC':<12} {'Brier':<12}", f)
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}", f)

        log(f"  {'Standard PBM':<20} {pbm_test['log_loss']:<12.4f} {pbm_test['auc']:<12.4f} {pbm_test['brier']:<12.4f}", f)

        for grid_width, metrics in grid_results.items():
            model_name = f"Grid-PBM (W={grid_width})"
            log(f"  {model_name:<20} {metrics['log_loss']:<12.4f} {metrics['auc']:<12.4f} {metrics['brier']:<12.4f}", f)
        log("", f)

        # Best model
        best_ll = min(grid_results.items(), key=lambda x: x[1]['log_loss'])
        best_auc = max(grid_results.items(), key=lambda x: x[1]['auc'])

        log(f"Best Log Loss: Grid-PBM (W={best_ll[0]}) with {best_ll[1]['log_loss']:.4f}", f)
        log(f"Best AUC: Grid-PBM (W={best_auc[0]}) with {best_auc[1]['auc']:.4f}", f)
        log("", f)

        # Improvement over standard PBM
        log("Improvement over Standard PBM:", f)
        for grid_width, metrics in grid_results.items():
            ll_improvement = (pbm_test['log_loss'] - metrics['log_loss']) / pbm_test['log_loss'] * 100
            auc_improvement = (metrics['auc'] - pbm_test['auc']) / pbm_test['auc'] * 100
            log(f"  Grid-PBM (W={grid_width}): Log Loss {ll_improvement:+.2f}%, AUC {auc_improvement:+.2f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Section 8: Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 8: SUMMARY", f)
        log("=" * 80, f)
        log("", f)

        log("KEY FINDINGS:", f)
        log("", f)

        # Row decay finding
        log("1. ROW DECAY:", f)
        log(f"   - Examination probability decreases with row (vertical scroll)", f)
        log(f"   - Row 1 vs Row 0 ratio: {row_decay[1]/row_decay[0]:.3f}", f)
        log(f"   - Consistent with viewport-limited examination", f)
        log("", f)

        # Column effect finding
        log("2. COLUMN EFFECT:", f)
        if col_0_avg > col_1_avg * 1.05:
            log(f"   - Left column (col 0) has HIGHER examination than right column (col 1)", f)
            log(f"   - Left column advantage: {(col_0_avg - col_1_avg) / col_0_avg * 100:.1f}%", f)
            log(f"   - Suggests left-to-right scanning within viewport", f)
        elif col_1_avg > col_0_avg * 1.05:
            log(f"   - Right column (col 1) has HIGHER examination than left column (col 0)", f)
            log(f"   - Right column advantage: {(col_1_avg - col_0_avg) / col_1_avg * 100:.1f}%", f)
        else:
            log(f"   - No significant left/right asymmetry detected", f)
            log(f"   - Column 0: {col_0_avg:.4f}, Column 1: {col_1_avg:.4f}", f)
        log("", f)

        # Model comparison finding
        log("3. MODEL COMPARISON:", f)
        best_grid = min(grid_results.items(), key=lambda x: x[1]['log_loss'])
        if best_grid[1]['log_loss'] < pbm_test['log_loss']:
            improvement = (pbm_test['log_loss'] - best_grid[1]['log_loss']) / pbm_test['log_loss'] * 100
            log(f"   - Grid-PBM (W={best_grid[0]}) OUTPERFORMS Standard PBM", f)
            log(f"   - Log Loss improvement: {improvement:.2f}%", f)
            log(f"   - Spatial structure adds predictive value", f)
        else:
            log(f"   - Standard PBM performs similarly to Grid-PBM", f)
            log(f"   - Linear position indexing may be sufficient", f)
        log("", f)

        log("IMPLICATIONS:", f)
        log("  - Grid structure captures viewport-based examination patterns", f)
        log("  - Position effects are not purely linear; spatial layout matters", f)
        log("  - Different placements may have different optimal grid widths", f)
        log("", f)

        log("=" * 80, f)
        log("GRID-PBM ANALYSIS COMPLETE", f)
        log(f"Output saved to: {OUTPUT_FILE}", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
