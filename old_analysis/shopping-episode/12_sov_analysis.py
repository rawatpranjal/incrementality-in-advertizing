#!/usr/bin/env python3
"""
12_sov_analysis.py
Tests whether occupying multiple top slots in an auction yields super-linear returns.
Uses auction fixed effects to compare vendors within the same auction.
Tests nonlinearity: does moving from 1 to 2 slots yield more than 2x the effect of 1 slot?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyfixest as pf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "12_sov_analysis.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def safe_coef(model, var):
    """Safely extract coefficient."""
    try:
        coefs = model.coef()
        if var in coefs:
            return coefs[var]
    except:
        pass
    return None

def safe_se(model, var):
    """Safely extract SE."""
    try:
        ses = model.se()
        if var in ses:
            return ses[var]
    except:
        pass
    return None

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("12_SOV_ANALYSIS (Share of Voice)", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script tests whether occupying multiple top slots in an auction yields", f)
        log("super-linear returns. The hypothesis is that dominating the 'above the fold'", f)
        log("pixels creates a 'trust signal' or exhausts user patience, forcing selection.", f)
        log("", f)
        log("We use auction fixed effects to compare vendors within the same auction,", f)
        log("controlling for auction-level demand shocks. The key test is whether the", f)
        log("marginal effect of a 2nd or 3rd slot exceeds that of the 1st slot.", f)
        log("", f)
        log("Specifications:", f)
        log("  Model 1 (Linear): Click_av ~ n_slots | auction_id + vendor_id", f)
        log("  Model 2 (Categorical): Click_av ~ slots_1 + slots_2 + slots_3plus | FE", f)
        log("  Model 3 (Top-N): Click_av ~ n_slots_top3 + n_slots_top10 | FE", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("LOADING DATA", f)
        log("-" * 40, f)

        log("\nLoading auction results (bids)...", f)
        bids = pd.read_parquet(SOURCE_DIR / 'auctions_results_365d.parquet')
        log(f"  Bids (full): {len(bids):,}", f)

        # Sample 10% of auctions for faster processing
        unique_auctions = bids['AUCTION_ID'].unique()
        np.random.seed(42)
        sampled_auctions = np.random.choice(unique_auctions, size=len(unique_auctions)//10, replace=False)
        bids = bids[bids['AUCTION_ID'].isin(sampled_auctions)]
        log(f"  Bids (10% sample): {len(bids):,}", f)

        log("\nLoading auction_panel...", f)
        auction_panel = pd.read_parquet(DATA_DIR / 'auction_panel.parquet')
        log(f"  Auction-vendor pairs: {len(auction_panel):,}", f)

        log("\nLoading promoted_events (for clicks)...", f)
        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        log(f"  Clicks: {len(promoted_events):,}", f)

        log("\nLoading impressions...", f)
        impressions = pd.read_parquet(SOURCE_DIR / 'impressions_365d.parquet')
        log(f"  Impressions: {len(impressions):,}", f)

        # ============================================================
        # 2. BUILD SOV METRICS FROM RAW BIDS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING SHARE OF VOICE METRICS", f)
        log("=" * 80, f)

        # Get K per auction from auction_panel
        k_per_auction = auction_panel.groupby('auction_id')['K'].first().to_dict()

        log("\nComputing slot counts per (auction, vendor)...", f)

        # Add K to bids
        bids['K'] = bids['AUCTION_ID'].map(k_per_auction)
        bids = bids.dropna(subset=['K'])
        bids['K'] = bids['K'].astype(int)

        # Create shown indicator
        bids['shown'] = bids['RANKING'] <= bids['K']

        # Compute SOV metrics per (auction, vendor)
        log("\nAggregating to (auction, vendor) level...", f)

        # Faster vectorized aggregation
        sov_agg = bids.groupby(['AUCTION_ID', 'VENDOR_ID']).agg({
            'RANKING': ['count', 'min', 'max', 'mean'],
            'shown': 'sum',
            'K': 'first'
        })
        sov_agg.columns = ['n_products', 'best_rank', 'worst_rank', 'avg_rank', 'n_shown', 'K']
        sov_agg = sov_agg.reset_index()
        sov_agg.columns = ['auction_id', 'vendor_id', 'n_products', 'best_rank', 'worst_rank', 'avg_rank', 'n_shown', 'K']
        sov_agg['rank_span'] = sov_agg['worst_rank'] - sov_agg['best_rank'] + 1

        # Top-N counts require merge back
        bids['top3'] = (bids['RANKING'] <= 3).astype(int)
        bids['top5'] = (bids['RANKING'] <= 5).astype(int)
        bids['top10'] = (bids['RANKING'] <= 10).astype(int)

        topn_agg = bids.groupby(['AUCTION_ID', 'VENDOR_ID']).agg({
            'top3': 'sum', 'top5': 'sum', 'top10': 'sum'
        }).reset_index()
        topn_agg.columns = ['auction_id', 'vendor_id', 'n_slots_top3', 'n_slots_top5', 'n_slots_top10']

        sov_panel = sov_agg.merge(topn_agg, on=['auction_id', 'vendor_id'])

        # Check for contiguity: if rank_span == n_products, then contiguous
        log("\nComputing contiguity...", f)
        sov_panel['contiguous'] = (sov_panel['rank_span'] == sov_panel['n_products']).astype(int)

        log(f"\nSOV panel: {len(sov_panel):,} (auction, vendor) pairs", f)

        # ============================================================
        # 3. ADD CLICK OUTCOMES
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ADDING CLICK OUTCOMES", f)
        log("=" * 80, f)

        # Aggregate clicks to (auction, vendor) level
        clicks_av = promoted_events.groupby(['auction_id', 'vendor_id']).size().reset_index(name='n_clicks')

        log(f"\nClick aggregates: {len(clicks_av):,} (auction, vendor) pairs with clicks", f)

        # Merge with SOV panel
        sov_panel = sov_panel.merge(clicks_av, on=['auction_id', 'vendor_id'], how='left')
        sov_panel['n_clicks'] = sov_panel['n_clicks'].fillna(0).astype(int)
        sov_panel['clicked'] = (sov_panel['n_clicks'] > 0).astype(int)

        log(f"  SOV panel with outcomes: {len(sov_panel):,}", f)
        log(f"  Pairs with clicks: {sov_panel['clicked'].sum():,} ({sov_panel['clicked'].mean()*100:.3f}%)", f)

        # ============================================================
        # 4. DESCRIPTIVE STATISTICS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("DESCRIPTIVE STATISTICS", f)
        log("=" * 80, f)

        log("\n--- Slot Distribution ---", f)
        slot_dist = sov_panel['n_products'].value_counts().sort_index()
        for n_slots in slot_dist.index[:10]:
            pct = slot_dist[n_slots] / len(sov_panel) * 100
            log(f"  {n_slots} slot(s): {slot_dist[n_slots]:,} ({pct:.2f}%)", f)

        log("\n--- Click Rate by Number of Slots ---", f)
        log(f"{'Slots':>8} {'N':>12} {'Clicks':>10} {'Rate':>10}", f)
        log("-" * 45, f)

        for n_slots in sorted(sov_panel['n_products'].unique())[:10]:
            subset = sov_panel[sov_panel['n_products'] == n_slots]
            click_rate = subset['clicked'].mean() * 100
            log(f"{n_slots:>8} {len(subset):>12,} {subset['clicked'].sum():>10,} {click_rate:>9.4f}%", f)

        log("\n--- Click Rate by Number of Top-3 Slots ---", f)
        for n_slots in sorted(sov_panel['n_slots_top3'].unique())[:5]:
            subset = sov_panel[sov_panel['n_slots_top3'] == n_slots]
            click_rate = subset['clicked'].mean() * 100
            log(f"  Top-3 slots = {n_slots}: {click_rate:.4f}% (n={len(subset):,})", f)

        log("\n--- Contiguity Effect ---", f)
        multi_slot = sov_panel[sov_panel['n_products'] >= 2]
        log(f"  Multi-slot vendors: {len(multi_slot):,}", f)
        if len(multi_slot) > 0:
            contig = multi_slot[multi_slot['contiguous'] == 1]
            non_contig = multi_slot[multi_slot['contiguous'] == 0]
            log(f"  Contiguous: {len(contig):,} ({len(contig)/len(multi_slot)*100:.1f}%), click rate = {contig['clicked'].mean()*100:.4f}%", f)
            log(f"  Non-contiguous: {len(non_contig):,} ({len(non_contig)/len(multi_slot)*100:.1f}%), click rate = {non_contig['clicked'].mean()*100:.4f}%", f)

        # ============================================================
        # 5. CREATE CATEGORICAL SLOT VARIABLES
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("PREPARING REGRESSION VARIABLES", f)
        log("=" * 80, f)

        # Categorical slots
        sov_panel['slots_0'] = (sov_panel['n_shown'] == 0).astype(int)
        sov_panel['slots_1'] = (sov_panel['n_shown'] == 1).astype(int)
        sov_panel['slots_2'] = (sov_panel['n_shown'] == 2).astype(int)
        sov_panel['slots_3plus'] = (sov_panel['n_shown'] >= 3).astype(int)

        log(f"\nSlot categories (shown slots only):", f)
        log(f"  0 shown: {sov_panel['slots_0'].sum():,}", f)
        log(f"  1 shown: {sov_panel['slots_1'].sum():,}", f)
        log(f"  2 shown: {sov_panel['slots_2'].sum():,}", f)
        log(f"  3+ shown: {sov_panel['slots_3plus'].sum():,}", f)

        # Filter to vendors with at least 1 shown slot (eligible to be clicked)
        sov_eligible = sov_panel[sov_panel['n_shown'] >= 1].copy()
        log(f"\nEligible vendors (n_shown >= 1): {len(sov_eligible):,}", f)

        # ============================================================
        # 6. REGRESSION ANALYSIS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("REGRESSION ANALYSIS", f)
        log("=" * 80, f)

        # Model 1: Linear effect of n_slots
        log("\n--- Model 1: Linear SOV Effect ---", f)
        log("Model: clicked ~ n_shown | auction_id + vendor_id", f)
        try:
            model1 = pf.feols("clicked ~ n_shown | auction_id + vendor_id",
                              data=sov_eligible, vcov={'CRV1': 'vendor_id'})
            log(str(model1.summary()), f)

            beta_linear = safe_coef(model1, 'n_shown')
            se_linear = safe_se(model1, 'n_shown')
            log(f"\n  β_n_shown = {beta_linear:.6f} (SE = {se_linear:.6f})", f)
            log(f"  Each additional shown slot increases click prob by {beta_linear*100:.4f} pp", f)

        except Exception as e:
            log(f"Model 1 failed: {e}", f)

        # Model 2: Categorical slots (testing nonlinearity)
        log("\n--- Model 2: Categorical Slots (Nonlinearity Test) ---", f)
        log("Model: clicked ~ slots_2 + slots_3plus | auction_id + vendor_id", f)
        log("(Base: slots_1 = 1 shown slot)", f)
        try:
            # Need to ensure we're comparing to 1-slot baseline
            sov_eligible_multi = sov_eligible[sov_eligible['n_shown'] >= 1].copy()

            model2 = pf.feols("clicked ~ slots_2 + slots_3plus | auction_id + vendor_id",
                              data=sov_eligible_multi, vcov={'CRV1': 'vendor_id'})
            log(str(model2.summary()), f)

            beta_2 = safe_coef(model2, 'slots_2')
            beta_3plus = safe_coef(model2, 'slots_3plus')

            if beta_2 and beta_3plus:
                log(f"\n  β_slots_2 = {beta_2:.6f} (2 slots vs 1 slot)", f)
                log(f"  β_slots_3plus = {beta_3plus:.6f} (3+ slots vs 1 slot)", f)

                # Test for super-linearity
                if beta_2 > 0 and beta_3plus > beta_2:
                    log("  -> SUPER-LINEAR: 3+ slots effect > 2 slots effect > 0", f)
                elif beta_2 > 0:
                    log("  -> POSITIVE: More slots help, but diminishing returns", f)
                else:
                    log("  -> NO EFFECT or NEGATIVE", f)

        except Exception as e:
            log(f"Model 2 failed: {e}", f)

        # Model 3: Top-N slots
        log("\n--- Model 3: Top-N Slot Effects ---", f)
        log("Model: clicked ~ n_slots_top3 + n_slots_top10 | auction_id + vendor_id", f)
        try:
            # Add n_slots_top10_only = top10 excluding top3
            sov_eligible['n_slots_top10_only'] = sov_eligible['n_slots_top10'] - sov_eligible['n_slots_top3']

            model3 = pf.feols("clicked ~ n_slots_top3 + n_slots_top10_only | auction_id + vendor_id",
                              data=sov_eligible, vcov={'CRV1': 'vendor_id'})
            log(str(model3.summary()), f)

            beta_top3 = safe_coef(model3, 'n_slots_top3')
            beta_top10 = safe_coef(model3, 'n_slots_top10_only')

            if beta_top3 and beta_top10:
                log(f"\n  β_top3 = {beta_top3:.6f} (slots in top 3)", f)
                log(f"  β_top10_only = {beta_top10:.6f} (slots in 4-10)", f)
                log(f"  Ratio: {beta_top3/beta_top10:.2f}x (top-3 vs 4-10)", f)

        except Exception as e:
            log(f"Model 3 failed: {e}", f)

        # Model 4: Contiguity effect (multi-slot only)
        log("\n--- Model 4: Contiguity Effect (Multi-slot only) ---", f)
        log("Model: clicked ~ n_shown + contiguous | auction_id + vendor_id", f)
        try:
            multi_slot_eligible = sov_eligible[sov_eligible['n_shown'] >= 2].copy()
            log(f"  Multi-slot sample: {len(multi_slot_eligible):,}", f)

            if len(multi_slot_eligible) > 100:
                model4 = pf.feols("clicked ~ n_shown + contiguous | auction_id + vendor_id",
                                  data=multi_slot_eligible, vcov={'CRV1': 'vendor_id'})
                log(str(model4.summary()), f)

                beta_contig = safe_coef(model4, 'contiguous')
                if beta_contig:
                    log(f"\n  β_contiguous = {beta_contig:.6f}", f)
                    if beta_contig > 0:
                        log("  -> POSITIVE: Contiguous slots (adjacent ranks) perform better", f)
                    else:
                        log("  -> NO EFFECT: Contiguity doesn't matter", f)
            else:
                log("  Insufficient multi-slot observations for regression", f)

        except Exception as e:
            log(f"Model 4 failed: {e}", f)

        # Model 5: Best rank control
        log("\n--- Model 5: SOV with Rank Control ---", f)
        log("Model: clicked ~ n_shown + best_rank | auction_id + vendor_id", f)
        try:
            model5 = pf.feols("clicked ~ n_shown + best_rank | auction_id + vendor_id",
                              data=sov_eligible, vcov={'CRV1': 'vendor_id'})
            log(str(model5.summary()), f)

            beta_n_shown = safe_coef(model5, 'n_shown')
            beta_rank = safe_coef(model5, 'best_rank')

            if beta_n_shown and beta_rank:
                log(f"\n  β_n_shown = {beta_n_shown:.6f} (controlling for best_rank)", f)
                log(f"  β_best_rank = {beta_rank:.6f} (lower rank = better)", f)

        except Exception as e:
            log(f"Model 5 failed: {e}", f)

        # ============================================================
        # 7. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SHARE OF VOICE ANALYSIS SUMMARY", f)
        log("=" * 80, f)

        log(f"\nData:", f)
        log(f"  Total (auction, vendor) pairs: {len(sov_panel):,}", f)
        log(f"  Eligible (shown >= 1): {len(sov_eligible):,}", f)
        log(f"  Multi-slot vendors: {(sov_eligible['n_shown'] >= 2).sum():,}", f)
        log(f"  Click rate: {sov_eligible['clicked'].mean()*100:.4f}%", f)

        log(f"\nClick Rate by Slot Count:", f)
        for n in [1, 2, 3]:
            subset = sov_eligible[sov_eligible['n_shown'] == n]
            if len(subset) > 0:
                log(f"  {n} slot(s): {subset['clicked'].mean()*100:.4f}% (n={len(subset):,})", f)

        log("\n" + "=" * 80, f)
        log("12_SOV_ANALYSIS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
