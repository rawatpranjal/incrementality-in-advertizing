#!/usr/bin/env python3
"""
Placement Interpretation Verification

Tests hypothesized placement interpretations using all available data signals
from both _all and _p5 datasets.

Hypotheses:
- Placement 1: Browse/Feed (81% delivery, 36s gaps, standard position decay)
- Placement 2: PDP Carousel (4% CTR, CTR increases with rank, multi-click)
- Placement 3: Search Pagination (2.9s gaps, 87% self-transition, 20% delivery)
- Placement 5: Bot/Scraper (150 users, 65 auctions/user, extreme concentration)

Output: results/10_placement_verification.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "10_placement_verification.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(f):
    """Load all parquet files for analysis."""
    log("=" * 80, f)
    log("LOADING DATA", f)
    log("=" * 80, f)

    datasets = {}

    # Load _all datasets
    all_files = {
        'au_all': 'auctions_users_all.parquet',
        'ar_all': 'auctions_results_all.parquet',
        'imp_all': 'impressions_all.parquet',
        'clicks_all': 'clicks_all.parquet',
        'catalog_all': 'catalog_all.parquet',
    }

    # Load _p5 datasets
    p5_files = {
        'au_p5': 'auctions_users_p5.parquet',
        'ar_p5': 'auctions_results_p5.parquet',
        'imp_p5': 'impressions_p5.parquet',
        'clicks_p5': 'clicks_p5.parquet',
        'catalog_p5': 'catalog_p5.parquet',
    }

    for name, filename in tqdm({**all_files, **p5_files}.items(), desc="Loading data"):
        filepath = DATA_DIR / filename
        if filepath.exists():
            datasets[name] = pd.read_parquet(filepath)
            log(f"  Loaded {name}: {len(datasets[name]):,} rows", f)
        else:
            log(f"  WARNING: {filename} not found", f)

    return datasets

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_placement_data(datasets, placement, suffix='all'):
    """Extract data for a specific placement from the dataset."""
    au = datasets[f'au_{suffix}']
    ar = datasets[f'ar_{suffix}']
    imp = datasets[f'imp_{suffix}']
    clicks = datasets[f'clicks_{suffix}']

    # Filter to placement
    au_p = au[au['PLACEMENT'] == str(placement)].copy()
    auction_ids = set(au_p['AUCTION_ID'])

    ar_p = ar[ar['AUCTION_ID'].isin(auction_ids)].copy()
    imp_p = imp[imp['AUCTION_ID'].isin(auction_ids)].copy()
    clicks_p = clicks[clicks['AUCTION_ID'].isin(auction_ids)].copy()

    return au_p, ar_p, imp_p, clicks_p

def verdict(passed, total, threshold=0.5):
    """Return verdict symbol based on pass rate."""
    if total == 0:
        return "?"
    rate = passed / total
    if rate >= threshold:
        return "✓"
    elif rate < 0.3:
        return "✗"
    else:
        return "?"

# =============================================================================
# PLACEMENT 5: BOT/SCRAPER DETECTION
# =============================================================================
def test_p5_bot_scraper(datasets, f):
    """Test battery for Placement 5: Bot/Scraper hypothesis."""
    log("\n" + "=" * 80, f)
    log("PLACEMENT 5: BOT/SCRAPER HYPOTHESIS", f)
    log("=" * 80, f)
    log("Claimed: ~150 users with 65 auctions/user, automated traffic", f)

    tests_passed = 0
    total_tests = 6

    # Get data for placement 5
    au_all, ar_all, imp_all, clicks_all = get_placement_data(datasets, 5, 'all')
    au_p5 = datasets.get('au_p5', au_all)
    ar_p5 = datasets.get('ar_p5', ar_all)
    imp_p5 = datasets.get('imp_p5', imp_all)
    clicks_p5 = datasets.get('clicks_p5', clicks_all)

    # -------------------------------------------------------------------------
    # Test 1: User Concentration
    # -------------------------------------------------------------------------
    log("\n--- Test 1: User Concentration ---", f)

    auctions_per_user_all = au_all.groupby('USER_ID').size()
    auctions_per_user_p5 = au_p5.groupby('USER_ID').size()

    n_users_all = len(auctions_per_user_all)
    n_users_p5 = len(auctions_per_user_p5)
    mean_auctions_all = auctions_per_user_all.mean()
    mean_auctions_p5 = auctions_per_user_p5.mean()
    max_auctions_all = auctions_per_user_all.max()
    max_auctions_p5 = auctions_per_user_p5.max()

    # Top 1% user share
    threshold_all = auctions_per_user_all.quantile(0.99)
    top1pct_users_all = auctions_per_user_all[auctions_per_user_all >= threshold_all]
    top1pct_share_all = top1pct_users_all.sum() / auctions_per_user_all.sum() * 100

    threshold_p5 = auctions_per_user_p5.quantile(0.99)
    top1pct_users_p5 = auctions_per_user_p5[auctions_per_user_p5 >= threshold_p5]
    top1pct_share_p5 = top1pct_users_p5.sum() / auctions_per_user_p5.sum() * 100

    log(f"  _all dataset:", f)
    log(f"    Users: {n_users_all:,}", f)
    log(f"    Mean auctions/user: {mean_auctions_all:.1f}", f)
    log(f"    Max auctions/user: {max_auctions_all:,}", f)
    log(f"    Top 1% users control: {top1pct_share_all:.1f}% of auctions", f)

    log(f"  _p5 dataset:", f)
    log(f"    Users: {n_users_p5:,}", f)
    log(f"    Mean auctions/user: {mean_auctions_p5:.1f}", f)
    log(f"    Max auctions/user: {max_auctions_p5:,}", f)
    log(f"    Top 1% users control: {top1pct_share_p5:.1f}% of auctions", f)

    # Pass if: mean auctions/user >> 10 (typical human is 1-3)
    test1_pass = mean_auctions_all > 30
    log(f"  Verdict: {'✓ PASS' if test1_pass else '✗ FAIL'} (mean auctions/user > 30 for bot-like behavior)", f)
    if test1_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 2: Hour-of-Day Activity
    # -------------------------------------------------------------------------
    log("\n--- Test 2: Hour-of-Day Activity ---", f)

    au_all['CREATED_AT'] = pd.to_datetime(au_all['CREATED_AT'])
    au_all['HOUR'] = au_all['CREATED_AT'].dt.hour

    hour_dist = au_all['HOUR'].value_counts().sort_index()
    hour_cv = hour_dist.std() / hour_dist.mean()  # Coefficient of variation

    log(f"  Hour distribution (Placement 5 from _all):", f)
    log(f"    Coefficient of variation: {hour_cv:.3f}", f)
    log(f"    Min hour count: {hour_dist.min():,}", f)
    log(f"    Max hour count: {hour_dist.max():,}", f)

    # Show hour distribution
    log(f"    Hour distribution histogram:", f)
    for hour in range(24):
        count = hour_dist.get(hour, 0)
        bar = '#' * int(count / hour_dist.max() * 30)
        log(f"      {hour:02d}:00  {bar} ({count:,})", f)

    # Pass if: low CV (flat distribution) OR concentrated in off-hours
    # Bots tend to have either very flat (24/7) or off-hours concentration
    test2_pass = hour_cv < 0.5  # Relatively flat distribution
    log(f"  Verdict: {'✓ PASS' if test2_pass else '✗ FAIL'} (CV < 0.5 suggests 24/7 automated activity)", f)
    if test2_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 3: Inter-Auction Regularity
    # -------------------------------------------------------------------------
    log("\n--- Test 3: Inter-Auction Regularity ---", f)

    au_sorted = au_all.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    gaps = au_sorted['TIME_GAP'].dropna()

    gap_mean = gaps.mean()
    gap_std = gaps.std()
    gap_cv = gap_std / gap_mean if gap_mean > 0 else np.inf

    log(f"  Time gap statistics (seconds):", f)
    log(f"    Mean: {gap_mean:.2f}", f)
    log(f"    Std: {gap_std:.2f}", f)
    log(f"    CV (std/mean): {gap_cv:.3f}", f)
    log(f"    Median: {gaps.median():.2f}", f)
    log(f"    P5: {gaps.quantile(0.05):.2f}", f)
    log(f"    P95: {gaps.quantile(0.95):.2f}", f)

    # Check for very regular intervals (scripted behavior)
    very_regular = (gaps < 5).sum() / len(gaps) * 100  # % under 5 seconds
    log(f"    Gaps < 5 seconds: {very_regular:.1f}%", f)

    # Pass if: low CV or high % of very short regular gaps
    test3_pass = gap_cv < 1.0 or very_regular > 50
    log(f"  Verdict: {'✓ PASS' if test3_pass else '✗ FAIL'} (low CV or >50% rapid-fire = scripted)", f)
    if test3_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 4: Purchase Rate
    # -------------------------------------------------------------------------
    log("\n--- Test 4: Purchase Rate ---", f)

    # Note: Purchases may not be in parquet files, check if available
    purchases_path = DATA_DIR / "purchases_all.parquet"
    if purchases_path.exists():
        purchases = pd.read_parquet(purchases_path)
        p5_users = set(au_all['USER_ID'])
        p5_purchasers = set(purchases['USER_ID']) & p5_users
        purchase_rate = len(p5_purchasers) / len(p5_users) * 100 if len(p5_users) > 0 else 0

        log(f"  P5 users: {len(p5_users):,}", f)
        log(f"  P5 users who purchased: {len(p5_purchasers):,}", f)
        log(f"  Purchase rate: {purchase_rate:.2f}%", f)

        test4_pass = purchase_rate < 5  # Bots don't purchase
    else:
        log(f"  PURCHASES data not available in parquet files", f)
        log(f"  Cannot verify purchase behavior (assuming inconclusive)", f)
        test4_pass = None  # Inconclusive

    if test4_pass is True:
        log(f"  Verdict: ✓ PASS (purchase rate < 5% = non-human)", f)
        tests_passed += 1
    elif test4_pass is False:
        log(f"  Verdict: ✗ FAIL (purchase rate >= 5%)", f)
    else:
        log(f"  Verdict: ? INCONCLUSIVE (data unavailable)", f)
        total_tests -= 1  # Don't count this test

    # -------------------------------------------------------------------------
    # Test 5: Click Behavior
    # -------------------------------------------------------------------------
    log("\n--- Test 5: Click Behavior ---", f)

    n_impressions = len(imp_all)
    n_clicks = len(clicks_all)
    ctr = n_clicks / n_impressions * 100 if n_impressions > 0 else 0

    # Compare to other placements
    au_all_full = datasets['au_all']
    imp_all_full = datasets['imp_all']
    clicks_all_full = datasets['clicks_all']

    log(f"  P5 CTR: {ctr:.3f}%", f)
    log(f"  P5 impressions: {n_impressions:,}", f)
    log(f"  P5 clicks: {n_clicks:,}", f)

    # Compare with overall CTR
    other_placements = ['1', '2', '3']
    for p in other_placements:
        au_p = au_all_full[au_all_full['PLACEMENT'] == p]
        auction_ids = set(au_p['AUCTION_ID'])
        imp_p = imp_all_full[imp_all_full['AUCTION_ID'].isin(auction_ids)]
        clicks_p = clicks_all_full[clicks_all_full['AUCTION_ID'].isin(auction_ids)]
        ctr_p = len(clicks_p) / len(imp_p) * 100 if len(imp_p) > 0 else 0
        log(f"  P{p} CTR for comparison: {ctr_p:.3f}%", f)

    # Pass if: CTR is extremely low or zero (bots don't click naturally)
    test5_pass = ctr < 1.0 or n_clicks < 100
    log(f"  Verdict: {'✓ PASS' if test5_pass else '✗ FAIL'} (CTR < 1% or very few clicks = bot)", f)
    if test5_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 6: Product Diversity
    # -------------------------------------------------------------------------
    log("\n--- Test 6: Product Diversity ---", f)

    # Check if same products appear repeatedly across auctions
    products_per_auction = ar_all.groupby('AUCTION_ID')['PRODUCT_ID'].nunique()
    total_unique_products = ar_all['PRODUCT_ID'].nunique()

    log(f"  Total unique products in P5: {total_unique_products:,}", f)
    log(f"  Products per auction: mean={products_per_auction.mean():.1f}, median={products_per_auction.median():.0f}", f)

    # Product concentration: how often do the same products appear?
    product_counts = ar_all['PRODUCT_ID'].value_counts()
    top10_products_share = product_counts.head(10).sum() / len(ar_all) * 100

    log(f"  Top 10 products account for: {top10_products_share:.1f}% of all bids", f)

    # Compare with other placements
    for p in other_placements[:2]:  # Just 1 and 2 for brevity
        au_p = au_all_full[au_all_full['PLACEMENT'] == p]
        auction_ids = set(au_p['AUCTION_ID'])
        ar_p = datasets['ar_all'][datasets['ar_all']['AUCTION_ID'].isin(auction_ids)]
        unique_prods = ar_p['PRODUCT_ID'].nunique()
        log(f"  P{p} unique products for comparison: {unique_prods:,}", f)

    # Pass if: product diversity is low relative to auction volume (systematic scraping)
    products_per_1000_auctions = total_unique_products / (len(au_all) / 1000) if len(au_all) > 0 else 0
    test6_pass = products_per_1000_auctions < 500  # Arbitrary threshold
    log(f"  Products per 1000 auctions: {products_per_1000_auctions:.1f}", f)
    log(f"  Verdict: {'✓ PASS' if test6_pass else '? INCONCLUSIVE'} (low diversity = systematic crawl)", f)
    if test6_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("\n--- PLACEMENT 5 SUMMARY ---", f)
    log(f"Tests passed: {tests_passed}/{total_tests}", f)

    if tests_passed >= 4:
        log(f"Verdict: ✓ CONFIRMED - Strong evidence for Bot/Scraper hypothesis", f)
    elif tests_passed >= 2:
        log(f"Verdict: ? INCONCLUSIVE - Mixed evidence", f)
    else:
        log(f"Verdict: ✗ REFUTED - Evidence does not support Bot/Scraper hypothesis", f)

    return tests_passed, total_tests

# =============================================================================
# PLACEMENT 3: SEARCH PAGINATION
# =============================================================================
def test_p3_search_pagination(datasets, f):
    """Test battery for Placement 3: Search Pagination hypothesis."""
    log("\n" + "=" * 80, f)
    log("PLACEMENT 3: SEARCH PAGINATION HYPOTHESIS", f)
    log("=" * 80, f)
    log("Claimed: ~3s gaps, ~87% self-transition, ~20% impression delivery", f)

    tests_passed = 0
    total_tests = 5

    au_all, ar_all, imp_all, clicks_all = get_placement_data(datasets, 3, 'all')

    # -------------------------------------------------------------------------
    # Test 1: Time Gaps (~3s expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 1: Time Gaps ---", f)

    au_all['CREATED_AT'] = pd.to_datetime(au_all['CREATED_AT'])
    au_sorted = au_all.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    gaps = au_sorted['TIME_GAP'].dropna()

    median_gap = gaps.median()
    mean_gap = gaps.mean()

    log(f"  Time gap statistics (seconds):", f)
    log(f"    Median: {median_gap:.2f}", f)
    log(f"    Mean: {mean_gap:.2f}", f)
    log(f"    P25: {gaps.quantile(0.25):.2f}", f)
    log(f"    P75: {gaps.quantile(0.75):.2f}", f)

    # Compare with claimed ~2.9s
    test1_pass = 1 < median_gap < 10  # Rapid but not instantaneous
    log(f"  Expected: ~2.9 seconds", f)
    log(f"  Verdict: {'✓ PASS' if test1_pass else '✗ FAIL'} (median gap between 1-10s)", f)
    if test1_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 2: Self-Transition Rate (~87% expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 2: Self-Transition Rate ---", f)

    # Need full dataset to compute transitions
    au_full = datasets['au_all'].copy()
    au_full['CREATED_AT'] = pd.to_datetime(au_full['CREATED_AT'])
    au_full = au_full.sort_values(['USER_ID', 'CREATED_AT'])
    au_full['NEXT_PLACEMENT'] = au_full.groupby('USER_ID')['PLACEMENT'].shift(-1)

    p3_transitions = au_full[au_full['PLACEMENT'] == '3']
    p3_to_p3 = (p3_transitions['NEXT_PLACEMENT'] == '3').sum()
    p3_total = p3_transitions['NEXT_PLACEMENT'].notna().sum()
    self_transition_rate = p3_to_p3 / p3_total * 100 if p3_total > 0 else 0

    log(f"  P3 -> P3 transitions: {p3_to_p3:,}", f)
    log(f"  Total P3 transitions: {p3_total:,}", f)
    log(f"  Self-transition rate: {self_transition_rate:.1f}%", f)

    # Show full transition matrix for P3
    log(f"\n  Transition probabilities from P3:", f)
    for next_p in sorted(au_full['PLACEMENT'].dropna().unique()):
        count = (p3_transitions['NEXT_PLACEMENT'] == next_p).sum()
        pct = count / p3_total * 100 if p3_total > 0 else 0
        log(f"    P3 -> P{next_p}: {pct:.1f}%", f)

    test2_pass = self_transition_rate > 70  # High self-transition
    log(f"  Expected: ~87%", f)
    log(f"  Verdict: {'✓ PASS' if test2_pass else '✗ FAIL'} (self-transition > 70%)", f)
    if test2_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 3: Session Bursts
    # -------------------------------------------------------------------------
    log("\n--- Test 3: Session Bursts ---", f)

    # Count consecutive P3 auctions per user
    au_sorted = au_full[au_full['PLACEMENT'] == '3'].sort_values(['USER_ID', 'CREATED_AT'])

    # Compute burst sizes (consecutive P3 with gaps < 30s)
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    au_sorted['NEW_BURST'] = (au_sorted['TIME_GAP'] > 30) | (au_sorted['TIME_GAP'].isna())
    au_sorted['BURST_ID'] = au_sorted.groupby('USER_ID')['NEW_BURST'].cumsum()

    burst_sizes = au_sorted.groupby(['USER_ID', 'BURST_ID']).size()

    log(f"  Burst size statistics:", f)
    log(f"    Mean burst size: {burst_sizes.mean():.1f}", f)
    log(f"    Median burst size: {burst_sizes.median():.0f}", f)
    log(f"    Max burst size: {burst_sizes.max()}", f)
    log(f"    Bursts with 5+ auctions: {(burst_sizes >= 5).sum():,} ({(burst_sizes >= 5).sum() / len(burst_sizes) * 100:.1f}%)", f)

    test3_pass = burst_sizes.mean() > 2  # Clustered activity
    log(f"  Verdict: {'✓ PASS' if test3_pass else '✗ FAIL'} (mean burst > 2 = paginated search)", f)
    if test3_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 4: Product Rank Shifts (same product at different ranks)
    # -------------------------------------------------------------------------
    log("\n--- Test 4: Product Rank Shifts ---", f)

    # Check if same products appear at different ranks across auctions
    product_ranks = ar_all.groupby('PRODUCT_ID')['RANKING'].agg(['nunique', 'min', 'max', 'count'])
    products_with_multiple_ranks = (product_ranks['nunique'] > 1).sum()
    total_products = len(product_ranks)

    log(f"  Products appearing at multiple ranks: {products_with_multiple_ranks:,} ({products_with_multiple_ranks / total_products * 100:.1f}%)", f)
    log(f"  Total unique products: {total_products:,}", f)

    # Products with large rank variance
    product_ranks['range'] = product_ranks['max'] - product_ranks['min']
    large_range = (product_ranks['range'] > 10).sum()
    log(f"  Products with rank range > 10: {large_range:,}", f)

    test4_pass = products_with_multiple_ranks / total_products > 0.1 if total_products > 0 else False
    log(f"  Verdict: {'✓ PASS' if test4_pass else '✗ FAIL'} (>10% products at multiple ranks = pagination)", f)
    if test4_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 5: Low Impression Delivery (~20% expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 5: Impression Delivery Rate ---", f)

    # Auctions with at least one impression
    auctions_with_imp = imp_all['AUCTION_ID'].nunique()
    total_auctions = au_all['AUCTION_ID'].nunique()
    delivery_rate = auctions_with_imp / total_auctions * 100 if total_auctions > 0 else 0

    # Winners vs impressions
    winners = ar_all[ar_all['IS_WINNER'] == True]
    n_winners = len(winners)
    n_impressions = len(imp_all)
    winner_delivery = n_impressions / n_winners * 100 if n_winners > 0 else 0

    log(f"  Auctions with impressions: {auctions_with_imp:,} / {total_auctions:,}", f)
    log(f"  Auction delivery rate: {delivery_rate:.1f}%", f)
    log(f"  Winners: {n_winners:,}", f)
    log(f"  Impressions: {n_impressions:,}", f)
    log(f"  Impressions per winner: {n_impressions / n_winners:.2f}" if n_winners > 0 else "  N/A", f)

    test5_pass = delivery_rate < 50  # Low delivery rate
    log(f"  Expected: ~20% delivery", f)
    log(f"  Verdict: {'✓ PASS' if test5_pass else '✗ FAIL'} (delivery < 50% = not all results viewed)", f)
    if test5_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("\n--- PLACEMENT 3 SUMMARY ---", f)
    log(f"Tests passed: {tests_passed}/{total_tests}", f)

    if tests_passed >= 4:
        log(f"Verdict: ✓ CONFIRMED - Strong evidence for Search Pagination hypothesis", f)
    elif tests_passed >= 2:
        log(f"Verdict: ? INCONCLUSIVE - Mixed evidence", f)
    else:
        log(f"Verdict: ✗ REFUTED - Evidence does not support Search Pagination hypothesis", f)

    return tests_passed, total_tests

# =============================================================================
# PLACEMENT 2: PDP CAROUSEL
# =============================================================================
def test_p2_pdp_carousel(datasets, f):
    """Test battery for Placement 2: PDP Carousel hypothesis."""
    log("\n" + "=" * 80, f)
    log("PLACEMENT 2: PDP CAROUSEL HYPOTHESIS", f)
    log("=" * 80, f)
    log("Claimed: ~4% CTR, CTR increases with rank, high multi-click rate", f)

    tests_passed = 0
    total_tests = 5

    au_all, ar_all, imp_all, clicks_all = get_placement_data(datasets, 2, 'all')

    # -------------------------------------------------------------------------
    # Test 1: CTR by Rank (flat or increasing expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 1: CTR by Rank ---", f)

    # Join impressions and clicks to auction results for rank
    imp_with_rank = imp_all.merge(
        ar_all[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )
    clicks_with_rank = clicks_all.merge(
        ar_all[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    log(f"  CTR by Rank:", f)
    log(f"  {'Rank':>6} {'Impressions':>12} {'Clicks':>10} {'CTR':>10}", f)
    log(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}", f)

    ctrs_by_rank = []
    for rank in sorted(imp_with_rank['RANKING'].dropna().unique())[:10]:
        n_imp = (imp_with_rank['RANKING'] == rank).sum()
        n_clicks = (clicks_with_rank['RANKING'] == rank).sum()
        ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
        ctrs_by_rank.append((rank, ctr))
        log(f"  {int(rank):>6} {n_imp:>12,} {n_clicks:>10,} {ctr:>9.2f}%", f)

    # Check if CTR is flat or increasing (NOT decreasing like typical browse)
    if len(ctrs_by_rank) >= 3:
        early_ctr = np.mean([c for r, c in ctrs_by_rank[:3]])
        late_ctr = np.mean([c for r, c in ctrs_by_rank[-3:]])
        ctr_trend = late_ctr - early_ctr
        log(f"\n  Early ranks (1-3) avg CTR: {early_ctr:.2f}%", f)
        log(f"  Late ranks avg CTR: {late_ctr:.2f}%", f)
        log(f"  Trend (late - early): {ctr_trend:+.2f}%", f)

        test1_pass = ctr_trend > -0.5  # Not strongly decreasing
        log(f"  Expected: flat or increasing CTR", f)
        log(f"  Verdict: {'✓ PASS' if test1_pass else '✗ FAIL'} (CTR not strongly decreasing)", f)
    else:
        test1_pass = False
        log(f"  Insufficient data for trend analysis", f)

    if test1_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 2: Category Homogeneity
    # -------------------------------------------------------------------------
    log("\n--- Test 2: Category Homogeneity ---", f)

    catalog = datasets['catalog_all']

    # Join auction results with catalog
    ar_with_cat = ar_all.merge(
        catalog[['PRODUCT_ID', 'CATEGORIES']],
        on='PRODUCT_ID',
        how='left'
    )

    # For each auction, check category overlap among products
    def compute_category_overlap(group):
        categories_list = group['CATEGORIES'].dropna().tolist()
        if len(categories_list) < 2:
            return np.nan

        # Flatten and find common categories
        all_cats = []
        for cats in categories_list:
            if isinstance(cats, (list, np.ndarray)):
                all_cats.extend(cats)

        if not all_cats:
            return np.nan

        from collections import Counter
        cat_counts = Counter(all_cats)
        most_common_freq = cat_counts.most_common(1)[0][1] if cat_counts else 0
        overlap_rate = most_common_freq / len(categories_list)
        return overlap_rate

    # Sample auctions for efficiency
    sample_auctions = ar_with_cat.groupby('AUCTION_ID').head(1)['AUCTION_ID'].sample(min(1000, len(ar_with_cat['AUCTION_ID'].unique())), random_state=42)
    ar_sample = ar_with_cat[ar_with_cat['AUCTION_ID'].isin(sample_auctions)]

    overlaps = []
    for auction_id, group in tqdm(ar_sample.groupby('AUCTION_ID'), desc="Computing category overlap", leave=False):
        overlap = compute_category_overlap(group)
        if not np.isnan(overlap):
            overlaps.append(overlap)

    if overlaps:
        mean_overlap = np.mean(overlaps)
        log(f"  Mean category overlap rate: {mean_overlap:.2f}", f)
        log(f"  Auctions analyzed: {len(overlaps):,}", f)
        test2_pass = mean_overlap > 0.3  # Moderate homogeneity
    else:
        log(f"  Could not compute category overlap", f)
        test2_pass = False

    log(f"  Expected: products share categories (PDP 'similar items')", f)
    log(f"  Verdict: {'✓ PASS' if test2_pass else '? INCONCLUSIVE'}", f)
    if test2_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 3: Multi-Click Rate
    # -------------------------------------------------------------------------
    log("\n--- Test 3: Multi-Click Rate ---", f)

    clicks_per_auction = clicks_all.groupby('AUCTION_ID').size()
    auctions_with_clicks = len(clicks_per_auction)
    multi_click_auctions = (clicks_per_auction >= 2).sum()
    multi_click_rate = multi_click_auctions / auctions_with_clicks * 100 if auctions_with_clicks > 0 else 0

    log(f"  Auctions with any click: {auctions_with_clicks:,}", f)
    log(f"  Auctions with 2+ clicks: {multi_click_auctions:,}", f)
    log(f"  Multi-click rate: {multi_click_rate:.1f}%", f)
    log(f"  Clicks per auction distribution:", f)
    log(f"    1 click: {(clicks_per_auction == 1).sum():,}", f)
    log(f"    2 clicks: {(clicks_per_auction == 2).sum():,}", f)
    log(f"    3+ clicks: {(clicks_per_auction >= 3).sum():,}", f)

    test3_pass = multi_click_rate > 5  # Higher multi-click than typical
    log(f"  Expected: high multi-click (carousel browsing)", f)
    log(f"  Verdict: {'✓ PASS' if test3_pass else '✗ FAIL'} (multi-click > 5%)", f)
    if test3_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 4: Transition Patterns
    # -------------------------------------------------------------------------
    log("\n--- Test 4: Transition Patterns ---", f)

    au_full = datasets['au_all'].copy()
    au_full['CREATED_AT'] = pd.to_datetime(au_full['CREATED_AT'])
    au_full = au_full.sort_values(['USER_ID', 'CREATED_AT'])
    au_full['PREV_PLACEMENT'] = au_full.groupby('USER_ID')['PLACEMENT'].shift(1)

    # What leads to P2?
    to_p2 = au_full[au_full['PLACEMENT'] == '2']
    from_placements = to_p2['PREV_PLACEMENT'].value_counts()
    total_to_p2 = to_p2['PREV_PLACEMENT'].notna().sum()

    log(f"  Transitions TO P2:", f)
    for p in sorted(from_placements.index):
        pct = from_placements[p] / total_to_p2 * 100 if total_to_p2 > 0 else 0
        log(f"    P{p} -> P2: {pct:.1f}%", f)

    # What follows P2?
    au_full['NEXT_PLACEMENT'] = au_full.groupby('USER_ID')['PLACEMENT'].shift(-1)
    from_p2 = au_full[au_full['PLACEMENT'] == '2']
    to_placements = from_p2['NEXT_PLACEMENT'].value_counts()
    total_from_p2 = from_p2['NEXT_PLACEMENT'].notna().sum()

    log(f"\n  Transitions FROM P2:", f)
    for p in sorted(to_placements.index):
        pct = to_placements[p] / total_from_p2 * 100 if total_from_p2 > 0 else 0
        log(f"    P2 -> P{p}: {pct:.1f}%", f)

    # P2 should come from P3 (search -> PDP) or P1 (browse -> PDP)
    from_search_or_browse = (from_placements.get('1', 0) + from_placements.get('3', 0)) / total_to_p2 * 100 if total_to_p2 > 0 else 0
    test4_pass = from_search_or_browse > 30  # Comes from product discovery
    log(f"\n  Expected: P2 follows P1/P3 (PDP after discovery)", f)
    log(f"  From P1 or P3: {from_search_or_browse:.1f}%", f)
    log(f"  Verdict: {'✓ PASS' if test4_pass else '? INCONCLUSIVE'}", f)
    if test4_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 5: High Overall CTR (~4% expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 5: Overall CTR ---", f)

    n_impressions = len(imp_all)
    n_clicks = len(clicks_all)
    overall_ctr = n_clicks / n_impressions * 100 if n_impressions > 0 else 0

    log(f"  Impressions: {n_impressions:,}", f)
    log(f"  Clicks: {n_clicks:,}", f)
    log(f"  CTR: {overall_ctr:.2f}%", f)

    # Compare with other placements
    log(f"\n  CTR comparison with other placements:", f)
    for p in ['1', '3', '5']:
        _, _, imp_p, clicks_p = get_placement_data(datasets, p, 'all')
        ctr_p = len(clicks_p) / len(imp_p) * 100 if len(imp_p) > 0 else 0
        log(f"    P{p} CTR: {ctr_p:.2f}%", f)

    test5_pass = overall_ctr > 2.5  # Higher than typical
    log(f"\n  Expected: ~4% (highest CTR)", f)
    log(f"  Verdict: {'✓ PASS' if test5_pass else '✗ FAIL'} (CTR > 2.5%)", f)
    if test5_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("\n--- PLACEMENT 2 SUMMARY ---", f)
    log(f"Tests passed: {tests_passed}/{total_tests}", f)

    if tests_passed >= 4:
        log(f"Verdict: ✓ CONFIRMED - Strong evidence for PDP Carousel hypothesis", f)
    elif tests_passed >= 2:
        log(f"Verdict: ? INCONCLUSIVE - Mixed evidence", f)
    else:
        log(f"Verdict: ✗ REFUTED - Evidence does not support PDP Carousel hypothesis", f)

    return tests_passed, total_tests

# =============================================================================
# PLACEMENT 1: BROWSE/FEED
# =============================================================================
def test_p1_browse_feed(datasets, f):
    """Test battery for Placement 1: Browse/Feed hypothesis."""
    log("\n" + "=" * 80, f)
    log("PLACEMENT 1: BROWSE/FEED HYPOTHESIS", f)
    log("=" * 80, f)
    log("Claimed: ~81% impression delivery, ~36s gaps, standard position decay", f)

    tests_passed = 0
    total_tests = 5

    au_all, ar_all, imp_all, clicks_all = get_placement_data(datasets, 1, 'all')

    # -------------------------------------------------------------------------
    # Test 1: Impression Delivery (~81% expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 1: Impression Delivery ---", f)

    auctions_with_imp = imp_all['AUCTION_ID'].nunique()
    total_auctions = au_all['AUCTION_ID'].nunique()
    delivery_rate = auctions_with_imp / total_auctions * 100 if total_auctions > 0 else 0

    winners = ar_all[ar_all['IS_WINNER'] == True]
    n_winners = len(winners)
    n_impressions = len(imp_all)

    log(f"  Auctions: {total_auctions:,}", f)
    log(f"  Auctions with impressions: {auctions_with_imp:,}", f)
    log(f"  Delivery rate: {delivery_rate:.1f}%", f)
    log(f"  Winners: {n_winners:,}", f)
    log(f"  Impressions: {n_impressions:,}", f)

    test1_pass = delivery_rate > 60  # High delivery
    log(f"  Expected: ~81%", f)
    log(f"  Verdict: {'✓ PASS' if test1_pass else '✗ FAIL'} (delivery > 60%)", f)
    if test1_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 2: CTR by Rank (modest decline expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 2: CTR by Rank ---", f)

    imp_with_rank = imp_all.merge(
        ar_all[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )
    clicks_with_rank = clicks_all.merge(
        ar_all[['AUCTION_ID', 'PRODUCT_ID', 'RANKING']],
        on=['AUCTION_ID', 'PRODUCT_ID'],
        how='left'
    )

    log(f"  {'Rank':>6} {'Impressions':>12} {'Clicks':>10} {'CTR':>10}", f)
    log(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}", f)

    ctrs_by_rank = []
    for rank in sorted(imp_with_rank['RANKING'].dropna().unique())[:10]:
        n_imp = (imp_with_rank['RANKING'] == rank).sum()
        n_clicks = (clicks_with_rank['RANKING'] == rank).sum()
        ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
        ctrs_by_rank.append((rank, ctr))
        log(f"  {int(rank):>6} {n_imp:>12,} {n_clicks:>10,} {ctr:>9.2f}%", f)

    # Check for declining CTR (standard position decay)
    if len(ctrs_by_rank) >= 3:
        early_ctr = np.mean([c for r, c in ctrs_by_rank[:2]])
        late_ctr = np.mean([c for r, c in ctrs_by_rank[2:5] if len(ctrs_by_rank) > 4])
        if not late_ctr:
            late_ctr = ctrs_by_rank[-1][1] if ctrs_by_rank else 0
        ctr_trend = late_ctr - early_ctr
        log(f"\n  Rank 1-2 avg CTR: {early_ctr:.2f}%", f)
        log(f"  Rank 3-5 avg CTR: {late_ctr:.2f}%", f)
        log(f"  Trend: {ctr_trend:+.2f}%", f)

        test2_pass = ctr_trend < 0  # Declining CTR is standard
    else:
        test2_pass = False

    log(f"  Expected: declining CTR (standard position decay)", f)
    log(f"  Verdict: {'✓ PASS' if test2_pass else '? INCONCLUSIVE'}", f)
    if test2_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 3: Time Gaps (~36s expected)
    # -------------------------------------------------------------------------
    log("\n--- Test 3: Time Gaps ---", f)

    au_all['CREATED_AT'] = pd.to_datetime(au_all['CREATED_AT'])
    au_sorted = au_all.sort_values(['USER_ID', 'CREATED_AT'])
    au_sorted['TIME_GAP'] = au_sorted.groupby('USER_ID')['CREATED_AT'].diff().dt.total_seconds()
    gaps = au_sorted['TIME_GAP'].dropna()

    median_gap = gaps.median()
    mean_gap = gaps.mean()

    log(f"  Time gap statistics (seconds):", f)
    log(f"    Median: {median_gap:.2f}", f)
    log(f"    Mean: {mean_gap:.2f}", f)
    log(f"    P25: {gaps.quantile(0.25):.2f}", f)
    log(f"    P75: {gaps.quantile(0.75):.2f}", f)

    test3_pass = median_gap > 20  # Leisurely browsing
    log(f"  Expected: ~36s (leisurely browsing)", f)
    log(f"  Verdict: {'✓ PASS' if test3_pass else '✗ FAIL'} (median > 20s)", f)
    if test3_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 4: Session Starts (often first placement)
    # -------------------------------------------------------------------------
    log("\n--- Test 4: Session Starts ---", f)

    au_full = datasets['au_all'].copy()
    au_full['CREATED_AT'] = pd.to_datetime(au_full['CREATED_AT'])
    au_full = au_full.sort_values(['USER_ID', 'CREATED_AT'])

    # First auction per user
    first_auctions = au_full.groupby('USER_ID').first()
    first_placement_dist = first_auctions['PLACEMENT'].value_counts(normalize=True) * 100

    log(f"  First placement per user session:", f)
    for p in sorted(first_placement_dist.index):
        log(f"    P{p}: {first_placement_dist[p]:.1f}%", f)

    p1_first = first_placement_dist.get('1', 0)
    test4_pass = p1_first > 15  # Reasonable share of session starts
    log(f"\n  Expected: P1 often starts sessions (browse entry)", f)
    log(f"  Verdict: {'✓ PASS' if test4_pass else '✗ FAIL'} (P1 starts > 15% of sessions)", f)
    if test4_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Test 5: User Distribution (normal, not concentrated)
    # -------------------------------------------------------------------------
    log("\n--- Test 5: User Distribution ---", f)

    auctions_per_user = au_all.groupby('USER_ID').size()
    n_users = len(auctions_per_user)
    mean_auctions = auctions_per_user.mean()
    max_auctions = auctions_per_user.max()

    # Check concentration
    top1pct_threshold = auctions_per_user.quantile(0.99)
    top1pct_share = auctions_per_user[auctions_per_user >= top1pct_threshold].sum() / auctions_per_user.sum() * 100

    log(f"  Users: {n_users:,}", f)
    log(f"  Mean auctions/user: {mean_auctions:.1f}", f)
    log(f"  Max auctions/user: {max_auctions}", f)
    log(f"  Top 1% users control: {top1pct_share:.1f}% of auctions", f)

    log(f"\n  Auctions per user distribution:", f)
    for pct in [50, 75, 90, 95, 99]:
        val = auctions_per_user.quantile(pct/100)
        log(f"    P{pct}: {val:.0f}", f)

    test5_pass = top1pct_share < 30  # Not highly concentrated
    log(f"\n  Expected: normal distribution (not concentrated)", f)
    log(f"  Verdict: {'✓ PASS' if test5_pass else '✗ FAIL'} (top 1% < 30% share)", f)
    if test5_pass:
        tests_passed += 1

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    log("\n--- PLACEMENT 1 SUMMARY ---", f)
    log(f"Tests passed: {tests_passed}/{total_tests}", f)

    if tests_passed >= 4:
        log(f"Verdict: ✓ CONFIRMED - Strong evidence for Browse/Feed hypothesis", f)
    elif tests_passed >= 2:
        log(f"Verdict: ? INCONCLUSIVE - Mixed evidence", f)
    else:
        log(f"Verdict: ✗ REFUTED - Evidence does not support Browse/Feed hypothesis", f)

    return tests_passed, total_tests

# =============================================================================
# MAIN
# =============================================================================
def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("PLACEMENT INTERPRETATION VERIFICATION", f)
        log("=" * 80, f)
        log("Purpose: Verify hypothesized placement interpretations using data signals", f)
        log(f"Data source: {DATA_DIR}", f)
        log("", f)

        # Load all data
        datasets = load_data(f)

        if not datasets:
            log("ERROR: No data loaded. Exiting.", f)
            return

        # Run test batteries
        results = {}

        # Placement 5: Bot/Scraper
        p5_passed, p5_total = test_p5_bot_scraper(datasets, f)
        results['5'] = ('Bot/Scraper', p5_passed, p5_total)

        # Placement 3: Search Pagination
        p3_passed, p3_total = test_p3_search_pagination(datasets, f)
        results['3'] = ('Search Pagination', p3_passed, p3_total)

        # Placement 2: PDP Carousel
        p2_passed, p2_total = test_p2_pdp_carousel(datasets, f)
        results['2'] = ('PDP Carousel', p2_passed, p2_total)

        # Placement 1: Browse/Feed
        p1_passed, p1_total = test_p1_browse_feed(datasets, f)
        results['1'] = ('Browse/Feed', p1_passed, p1_total)

        # =============================================================================
        # FINAL SUMMARY
        # =============================================================================
        log("\n" + "=" * 80, f)
        log("VERIFICATION SUMMARY", f)
        log("=" * 80, f)

        log(f"\n{'Placement':<12} {'Hypothesis':<20} {'Tests Passed':<15} {'Verdict':<12}", f)
        log(f"{'-'*12} {'-'*20} {'-'*15} {'-'*12}", f)

        for p in ['1', '2', '3', '5']:
            hypothesis, passed, total = results[p]
            rate = passed / total if total > 0 else 0
            if rate >= 0.7:
                v = "✓ CONFIRMED"
            elif rate >= 0.4:
                v = "? MIXED"
            else:
                v = "✗ REFUTED"
            log(f"P{p:<11} {hypothesis:<20} {passed}/{total:<13} {v:<12}", f)

        log("\n" + "=" * 80, f)
        log("INTERPRETATION KEY", f)
        log("=" * 80, f)
        log("✓ CONFIRMED: >=70% of tests passed - Strong evidence supports hypothesis", f)
        log("? MIXED: 40-70% of tests passed - Inconclusive, needs more investigation", f)
        log("✗ REFUTED: <40% of tests passed - Evidence contradicts hypothesis", f)

        log("\n" + "=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)
        log(f"Output written to: {OUTPUT_FILE}", f)

if __name__ == "__main__":
    main()
