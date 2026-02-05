"""
Generate naive CTR by rank figures for Placement 1 (Search) and Placement 3 (Product Page).
Illustrates the endogeneity problem: raw CTR declines with rank.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path("/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/0_data/round1")
OUTPUT_PATH = Path("/Users/pranjal/Code/topsort-incrementality/paper/figures/naive_ctr_by_rank.png")

def main():
    print("Loading data...")

    # Load parquet files
    auctions_results = pd.read_parquet(DATA_DIR / "auctions_results_all.parquet")
    auctions_users = pd.read_parquet(DATA_DIR / "auctions_users_all.parquet")
    impressions = pd.read_parquet(DATA_DIR / "impressions_all.parquet")
    clicks = pd.read_parquet(DATA_DIR / "clicks_all.parquet")

    print(f"Auctions results: {len(auctions_results):,} rows")
    print(f"Auctions users: {len(auctions_users):,} rows")
    print(f"Impressions: {len(impressions):,} rows")
    print(f"Clicks: {len(clicks):,} rows")

    # Join auctions_results with auctions_users to get placement
    print("Joining auctions with placements...")
    auctions_results = auctions_results.merge(
        auctions_users[['AUCTION_ID', 'PLACEMENT']],
        on='AUCTION_ID',
        how='left'
    )

    # Join impressions with auctions_results to get rank and placement
    print("Joining impressions with auction data...")
    impressions_with_rank = impressions.merge(
        auctions_results[['AUCTION_ID', 'PRODUCT_ID', 'RANKING', 'PLACEMENT']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    # Mark clicks
    clicks_set = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))
    impressions_with_rank['clicked'] = impressions_with_rank.apply(
        lambda r: (r['AUCTION_ID'], r['PRODUCT_ID']) in clicks_set, axis=1
    )

    # Compute CTR by rank for each placement
    def compute_ctr_by_rank(df, placement, max_rank=10):
        subset = df[(df['PLACEMENT'] == str(placement)) & (df['RANKING'] <= max_rank)]
        grouped = subset.groupby('RANKING').agg(
            impressions=('clicked', 'count'),
            clicks=('clicked', 'sum')
        ).reset_index()
        grouped['ctr'] = grouped['clicks'] / grouped['impressions'] * 100
        return grouped

    print("Computing CTR by rank for Placement 1 and Placement 3...")
    p1_ctr = compute_ctr_by_rank(impressions_with_rank, '1', max_rank=10)
    p3_ctr = compute_ctr_by_rank(impressions_with_rank, '3', max_rank=10)

    print("\nPlacement 1 (Search):")
    print(p1_ctr.to_string(index=False))
    print("\nPlacement 3 (Product Page):")
    print(p3_ctr.to_string(index=False))

    # Create figure
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Style settings
    bar_color = '#2c3e50'

    # Placement 1 (Search)
    ax1 = axes[0]
    ax1.bar(p1_ctr['RANKING'], p1_ctr['ctr'], color=bar_color, width=0.7)
    ax1.set_xlabel('Ad Rank', fontsize=11)
    ax1.set_ylabel('Click-Through Rate (%)', fontsize=11)
    ax1.set_title('Search (Placement 1)', fontsize=12)
    ax1.set_xticks(range(1, 11))
    ax1.set_ylim(0, max(p1_ctr['ctr'].max(), p3_ctr['ctr'].max()) * 1.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Placement 3 (Product Page)
    ax2 = axes[1]
    ax2.bar(p3_ctr['RANKING'], p3_ctr['ctr'], color=bar_color, width=0.7)
    ax2.set_xlabel('Ad Rank', fontsize=11)
    ax2.set_ylabel('Click-Through Rate (%)', fontsize=11)
    ax2.set_title('Product Page (Placement 3)', fontsize=12)
    ax2.set_xticks(range(1, 11))
    ax2.set_ylim(0, max(p1_ctr['ctr'].max(), p3_ctr['ctr'].max()) * 1.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
