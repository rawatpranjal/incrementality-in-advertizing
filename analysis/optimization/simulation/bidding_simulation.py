"""
Lift-Based Bidding Simulation
Demonstrates value of causal inference in auction-based advertising
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class User:
    """User with heterogeneous treatment effects"""
    user_id: int
    baseline_prob: float  # p0: baseline purchase probability
    true_lift: float      # tau: true causal effect
    ctr: float           # click-through rate
    clearing_price: float # auction clearing price

    @property
    def naive_cvr(self) -> float:
        """Observed CVR conflates baseline and lift"""
        return self.baseline_prob + self.true_lift

    @property
    def true_incremental_value(self) -> float:
        """True incremental conversions per impression"""
        return self.ctr * self.true_lift


def generate_synthetic_users(n_users: int, seed: int = 42) -> pd.DataFrame:
    """Generate users with negative correlation between baseline and lift"""
    np.random.seed(seed)

    # Baseline purchase probability (high for some users)
    baseline_prob = np.random.beta(2, 8, n_users)

    # True lift DECREASES as baseline increases (key insight)
    # High-intent users have low incrementality
    true_lift = 0.05 / (1 + 5 * baseline_prob) + np.random.normal(0, 0.005, n_users)
    true_lift = np.clip(true_lift, 0.001, 0.1)

    # CTR varies independently
    ctr = np.random.beta(3, 20, n_users)

    # Clearing prices from second-price auctions
    clearing_price = np.random.exponential(1.0, n_users) + 0.1

    users_df = pd.DataFrame({
        'user_id': range(n_users),
        'baseline_prob': baseline_prob,
        'true_lift': true_lift,
        'ctr': ctr,
        'clearing_price': clearing_price,
        'naive_cvr': baseline_prob + true_lift,
        'true_incremental': ctr * true_lift
    })

    return users_df


def compute_bid_scores(users_df: pd.DataFrame, strategy: str, noise_level: float = 0.3) -> np.ndarray:
    """Compute bid scores for different strategies"""

    if strategy == 'correlation':
        # Standard eCPM: CTR × CVR (includes baseline)
        scores = users_df['ctr'] * users_df['naive_cvr']

    elif strategy == 'ate':
        # Average treatment effect for all
        avg_lift = users_df['true_lift'].mean()
        scores = users_df['ctr'] * avg_lift

    elif strategy == 'hte':
        # Heterogeneous treatment effects with estimation noise
        estimated_lift = users_df['true_lift'] + np.random.normal(0, noise_level * users_df['true_lift'].std(), len(users_df))
        estimated_lift = np.clip(estimated_lift, 0.001, 0.2)
        scores = users_df['ctr'] * estimated_lift

    elif strategy == 'oracle':
        # Perfect knowledge of true lift
        scores = users_df['true_incremental']

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return scores


def run_auction(users_df: pd.DataFrame, scores: np.ndarray, budget: float, value_per_conversion: float = 50) -> Dict:
    """Run second-price auction with budget constraint"""

    # Compute bid amounts (score × value / lambda)
    # Start with lambda=1, could optimize later
    bids = scores * value_per_conversion

    # Sort by bid/price ratio (efficiency)
    efficiency = bids / users_df['clearing_price']
    sorted_idx = np.argsort(-efficiency)

    # Select impressions until budget exhausted
    cumulative_spend = 0
    selected_users = []

    for idx in sorted_idx:
        price = users_df.iloc[idx]['clearing_price']
        if cumulative_spend + price <= budget:
            cumulative_spend += price
            selected_users.append(idx)
        if cumulative_spend >= budget:
            break

    # Compute metrics for selected users
    if len(selected_users) == 0:
        return {
            'impressions': 0,
            'spend': 0,
            'incremental_conversions': 0,
            'total_conversions': 0,
            'iroas': 0,
            'icpa': float('inf'),
            'avg_baseline': 0,
            'avg_lift': 0
        }

    selected_df = users_df.iloc[selected_users]

    incremental_conversions = selected_df['true_incremental'].sum()
    total_conversions = (selected_df['ctr'] * selected_df['naive_cvr']).sum()
    incremental_value = incremental_conversions * value_per_conversion

    return {
        'impressions': len(selected_users),
        'spend': cumulative_spend,
        'incremental_conversions': incremental_conversions,
        'total_conversions': total_conversions,
        'iroas': incremental_value / cumulative_spend if cumulative_spend > 0 else 0,
        'icpa': cumulative_spend / incremental_conversions if incremental_conversions > 0 else float('inf'),
        'avg_baseline': selected_df['baseline_prob'].mean(),
        'avg_lift': selected_df['true_lift'].mean(),
        'correlation_with_baseline': np.corrcoef(selected_df['baseline_prob'], np.ones(len(selected_df)))[0,1] if len(selected_df) > 1 else 0,
        'correlation_with_lift': np.corrcoef(selected_df['true_lift'], np.ones(len(selected_df)))[0,1] if len(selected_df) > 1 else 0
    }


def run_budget_sweep(users_df: pd.DataFrame, budget_fractions: list = [0.1, 0.25, 0.5, 0.75, 1.0]) -> pd.DataFrame:
    """Compare strategies across different budget levels"""

    max_budget = users_df['clearing_price'].sum() * 0.3  # Can afford ~30% of all impressions
    strategies = ['correlation', 'ate', 'hte', 'oracle']

    results = []

    for budget_frac in tqdm(budget_fractions, desc="Budget levels"):
        budget = max_budget * budget_frac

        for strategy in strategies:
            scores = compute_bid_scores(users_df, strategy)
            metrics = run_auction(users_df, scores, budget)

            results.append({
                'budget_fraction': budget_frac,
                'budget': budget,
                'strategy': strategy,
                **metrics
            })

    return pd.DataFrame(results)


def compute_vendor_concentration(users_df: pd.DataFrame, selected_idx: list) -> float:
    """Compute HHI for vendor concentration"""
    # Simulate vendor assignment (could be based on product categories)
    np.random.seed(42)
    vendors = np.random.randint(0, 10, len(users_df))

    if len(selected_idx) == 0:
        return 0

    selected_vendors = vendors[selected_idx]
    vendor_counts = pd.Series(selected_vendors).value_counts()
    vendor_shares = vendor_counts / vendor_counts.sum()
    hhi = (vendor_shares ** 2).sum()

    return hhi


def main():
    """Run complete simulation and save results"""

    print("Generating synthetic user data...")
    users_df = generate_synthetic_users(10000)

    print("\nData summary:")
    print(f"Average baseline probability: {users_df['baseline_prob'].mean():.3f}")
    print(f"Average true lift: {users_df['true_lift'].mean():.3f}")
    print(f"Correlation(baseline, lift): {np.corrcoef(users_df['baseline_prob'], users_df['true_lift'])[0,1]:.3f}")

    print("\nRunning budget sweep simulation...")
    results_df = run_budget_sweep(users_df)

    # Save results
    results_df.to_csv('/Users/pranjal/Code/marketplace-incrementality/optimization/simulation/results/bidding_results.csv', index=False)

    # Print summary for main budget (50%)
    print("\n=== Results at 50% Budget ===")
    summary = results_df[results_df['budget_fraction'] == 0.5].set_index('strategy')

    for strategy in ['correlation', 'ate', 'hte', 'oracle']:
        if strategy in summary.index:
            row = summary.loc[strategy]
            print(f"\n{strategy.upper()}:")
            print(f"  Incremental Conversions: {row['incremental_conversions']:.1f}")
            print(f"  iROAS: {row['iroas']:.2f}")
            print(f"  iCPA: ${row['icpa']:.2f}")
            print(f"  Avg baseline of selected: {row['avg_baseline']:.3f}")
            print(f"  Avg lift of selected: {row['avg_lift']:.4f}")

    # Compute improvement ratios
    if 'correlation' in summary.index and 'hte' in summary.index:
        corr_iroas = summary.loc['correlation', 'iroas']
        hte_iroas = summary.loc['hte', 'iroas']
        improvement = (hte_iroas / corr_iroas - 1) * 100
        print(f"\n*** HTE improves iROAS by {improvement:.0f}% over correlation-based bidding ***")

    return results_df


if __name__ == "__main__":
    results = main()