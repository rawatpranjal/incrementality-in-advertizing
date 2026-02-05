"""
Incremental Slate Ranking Simulation
Demonstrates value of causal inference in promoted listings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Product:
    """Product with heterogeneous click and purchase effects"""
    product_id: int
    vendor_id: int
    organic_ctr: float      # Organic click rate
    organic_cvr: float      # Organic conversion rate (given click)
    rank_to_click_lift: float  # Incremental CTR from promotion
    click_to_purchase_lift: float  # Incremental CVR from ad click
    price: float
    cpc: float  # Cost per click

    @property
    def organic_purchase_rate(self) -> float:
        """Baseline purchase rate without promotion"""
        return self.organic_ctr * self.organic_cvr

    @property
    def incremental_value_per_slot(self) -> float:
        """True incremental value from promoting this product"""
        # Incremental clicks × incremental purchases per click × value
        return self.rank_to_click_lift * self.click_to_purchase_lift * self.price


def generate_search_session(n_products: int = 50, n_vendors: int = 10, seed: int = None) -> List[Product]:
    """Generate products for a search query"""
    if seed is not None:
        np.random.seed(seed)

    products = []

    for i in range(n_products):
        vendor_id = np.random.randint(0, n_vendors)

        # Organic metrics (what would happen without promotion)
        organic_ctr = np.random.beta(2, 20)  # ~10% baseline CTR
        organic_cvr = np.random.beta(2, 10)  # ~17% baseline CVR

        # Key insight: Popular products have LOWER incremental lift
        # (they would be clicked anyway)
        popularity = organic_ctr * organic_cvr

        # Incremental effects inversely related to popularity
        rank_to_click_lift = 0.15 / (1 + 10 * popularity) + np.random.normal(0, 0.01)
        rank_to_click_lift = np.clip(rank_to_click_lift, 0.001, 0.3)

        click_to_purchase_lift = 0.1 / (1 + 5 * organic_cvr) + np.random.normal(0, 0.01)
        click_to_purchase_lift = np.clip(click_to_purchase_lift, 0.001, 0.2)

        price = np.random.lognormal(3, 0.5)  # Products ~$20-50
        cpc = 0.05 * price + np.random.exponential(0.2)  # CPC related to price

        products.append(Product(
            product_id=i,
            vendor_id=vendor_id,
            organic_ctr=organic_ctr,
            organic_cvr=organic_cvr,
            rank_to_click_lift=rank_to_click_lift,
            click_to_purchase_lift=click_to_purchase_lift,
            price=price,
            cpc=cpc
        ))

    return products


def compute_position_effects(n_slots: int = 5) -> np.ndarray:
    """Position bias: P_j = 1/j^0.7"""
    positions = np.arange(1, n_slots + 1)
    return 1.0 / (positions ** 0.7)


def rank_products(products: List[Product], strategy: str, n_slots: int = 5) -> List[int]:
    """Rank products according to different strategies"""

    if strategy == 'ecpm':
        # Standard eCPM ranking: CTR × CVR × value
        # Uses OBSERVED metrics (includes organic)
        scores = [
            (p.organic_ctr + p.rank_to_click_lift) *
            (p.organic_cvr + p.click_to_purchase_lift) *
            p.price
            for p in products
        ]

    elif strategy == 'incremental':
        # Incremental ranking: only incremental value
        scores = [p.incremental_value_per_slot for p in products]

    elif strategy == 'random':
        # Random selection (baseline)
        scores = np.random.random(len(products))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Sort and return top N indices
    sorted_idx = np.argsort(scores)[::-1]
    return sorted_idx[:n_slots].tolist()


def simulate_slate_outcomes(
    products: List[Product],
    selected_idx: List[int],
    position_effects: np.ndarray,
    cannibalization_rate: float = 0.6
) -> Dict:
    """Simulate outcomes for a promoted slate"""

    if len(selected_idx) == 0:
        return {
            'promoted_clicks': 0,
            'organic_clicks': 0,
            'incremental_clicks': 0,
            'promoted_purchases': 0,
            'organic_purchases': 0,
            'incremental_purchases': 0,
            'spend': 0,
            'incremental_gmv': 0
        }

    promoted_clicks = 0
    incremental_clicks = 0
    promoted_purchases = 0
    incremental_purchases = 0
    spend = 0
    incremental_gmv = 0

    for slot, idx in enumerate(selected_idx):
        product = products[idx]
        position_effect = position_effects[slot]

        # Promoted slot gets position-boosted clicks
        slot_ctr = (product.organic_ctr + product.rank_to_click_lift) * position_effect
        promoted_clicks += slot_ctr

        # Only the lift portion is truly incremental
        slot_incremental_clicks = product.rank_to_click_lift * position_effect
        incremental_clicks += slot_incremental_clicks

        # Purchases from promoted clicks
        slot_purchases = slot_ctr * (product.organic_cvr + product.click_to_purchase_lift)
        promoted_purchases += slot_purchases

        # Truly incremental purchases (accounting for cannibalization)
        # Some promoted clicks would have happened organically
        cannibalized_clicks = slot_ctr * cannibalization_rate * product.organic_ctr / (product.organic_ctr + product.rank_to_click_lift)
        net_incremental_clicks = slot_ctr - cannibalized_clicks

        slot_incremental_purchases = net_incremental_clicks * product.click_to_purchase_lift
        incremental_purchases += slot_incremental_purchases

        # Cost and value
        spend += slot_ctr * product.cpc
        incremental_gmv += slot_incremental_purchases * product.price

    # Organic outcomes (non-promoted products)
    non_promoted_idx = [i for i in range(len(products)) if i not in selected_idx]
    organic_clicks = sum(products[i].organic_ctr for i in non_promoted_idx)
    organic_purchases = sum(products[i].organic_purchase_rate for i in non_promoted_idx)

    return {
        'promoted_clicks': promoted_clicks,
        'organic_clicks': organic_clicks,
        'incremental_clicks': incremental_clicks,
        'promoted_purchases': promoted_purchases,
        'organic_purchases': organic_purchases,
        'incremental_purchases': incremental_purchases,
        'spend': spend,
        'incremental_gmv': incremental_gmv,
        'iroas': incremental_gmv / spend if spend > 0 else 0
    }


def analyze_vendor_distribution(products: List[Product], selected_idx: List[int]) -> Dict:
    """Analyze how different strategies affect vendor mix"""

    if len(selected_idx) == 0:
        return {'n_unique_vendors': 0, 'hhi': 0, 'top_vendor_share': 0}

    selected_vendors = [products[i].vendor_id for i in selected_idx]
    vendor_counts = pd.Series(selected_vendors).value_counts()

    n_unique = len(vendor_counts)
    shares = vendor_counts / vendor_counts.sum()
    hhi = (shares ** 2).sum()
    top_share = shares.max()

    return {
        'n_unique_vendors': n_unique,
        'hhi': hhi,
        'top_vendor_share': top_share
    }


def run_session_simulation(n_sessions: int = 100, n_slots: int = 5) -> pd.DataFrame:
    """Run simulation across many search sessions"""

    strategies = ['random', 'ecpm', 'incremental']
    position_effects = compute_position_effects(n_slots)

    results = []

    for session_id in tqdm(range(n_sessions), desc="Simulating sessions"):
        products = generate_search_session(seed=session_id)

        for strategy in strategies:
            selected_idx = rank_products(products, strategy, n_slots)
            outcomes = simulate_slate_outcomes(products, selected_idx, position_effects)
            vendor_dist = analyze_vendor_distribution(products, selected_idx)

            results.append({
                'session_id': session_id,
                'strategy': strategy,
                **outcomes,
                **vendor_dist
            })

    return pd.DataFrame(results)


def main():
    """Run complete slate simulation"""

    print("Running slate ranking simulation...")
    results_df = run_session_simulation(n_sessions=100, n_slots=5)

    # Aggregate results by strategy
    summary = results_df.groupby('strategy').agg({
        'promoted_clicks': 'sum',
        'incremental_clicks': 'sum',
        'promoted_purchases': 'sum',
        'incremental_purchases': 'sum',
        'spend': 'sum',
        'incremental_gmv': 'sum',
        'n_unique_vendors': 'mean',
        'hhi': 'mean',
        'top_vendor_share': 'mean'
    }).round(2)

    # Compute iROAS
    summary['iroas'] = summary['incremental_gmv'] / summary['spend']
    summary['cannibalization_rate'] = 1 - (summary['incremental_purchases'] / summary['promoted_purchases'])

    # Save results
    results_df.to_csv('/Users/pranjal/Code/marketplace-incrementality/optimization/simulation/results/slate_results.csv', index=False)
    summary.to_csv('/Users/pranjal/Code/marketplace-incrementality/optimization/simulation/results/slate_summary.csv')

    print("\n=== Slate Ranking Results ===")
    print(summary[['incremental_purchases', 'incremental_gmv', 'spend', 'iroas', 'cannibalization_rate']])

    # Compute improvements
    if 'ecpm' in summary.index and 'incremental' in summary.index:
        ecpm_iroas = summary.loc['ecpm', 'iroas']
        inc_iroas = summary.loc['incremental', 'iroas']
        ecpm_cann = summary.loc['ecpm', 'cannibalization_rate']
        inc_cann = summary.loc['incremental', 'cannibalization_rate']

        print(f"\n*** Key Findings ***")
        print(f"Incremental ranking improves iROAS by {(inc_iroas/ecpm_iroas - 1)*100:.0f}% over eCPM")
        print(f"Cannibalization reduced from {ecpm_cann*100:.0f}% to {inc_cann*100:.0f}%")

    return results_df, summary


if __name__ == "__main__":
    results, summary = main()