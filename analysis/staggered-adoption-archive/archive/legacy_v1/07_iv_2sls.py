#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: IV/2SLS Estimation
Estimates the elasticity of GMV with respect to Ad Spend using instrumental variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyfixest as pf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"

OUTPUT_FILE = RESULTS_DIR / "07_iv_2sls.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("STAGGERED ADOPTION: IV/2SLS ESTIMATION", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Estimate the causal elasticity of GMV w.r.t. Ad Spend.", f)
        log("  Address endogeneity: Vendors spend more when they expect high sales.", f)
        log("", f)
        log("MODEL:", f)
        log("  Second Stage: ln(GMV_it) = beta * ln(Spend_hat_it) + mu_i + lambda_t + epsilon_it", f)
        log("  First Stage:  ln(Spend_it) = pi * Z_it + mu_i + lambda_t + nu_it", f)
        log("", f)
        log("INSTRUMENT:", f)
        log("  Z_it = Auction Competition Intensity", f)
        log("  Definition: Average number of competing bids in auctions for similar", f)
        log("  products/categories, excluding the focal vendor.", f)
        log("", f)
        log("  Relevance: High competition -> Higher CPC -> Exogenous spend variation", f)
        log("  Exclusion: Competitor behavior affects focal vendor's sales only", f)
        log("  through its impact on ad placement/cost, not directly.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            log("  Run 03_cohort_assignment.py first", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Construct instrument: Auction Competition Intensity
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("CONSTRUCTING INSTRUMENT", f)
        log("-" * 40, f)

        # Load auction data to compute competition
        auctions_path = RAW_DATA_DIR / "raw_sample_auctions_results.parquet"

        if auctions_path.exists():
            auctions = pd.read_parquet(auctions_path)
            auctions['CREATED_AT'] = pd.to_datetime(auctions['CREATED_AT'])
            auctions['week'] = auctions['CREATED_AT'].dt.to_period('W').apply(lambda x: x.start_time)

            log(f"  Loaded {len(auctions):,} auction results", f)

            # Calculate competition per auction
            auction_competition = auctions.groupby('AUCTION_ID').agg(
                n_bidders=('VENDOR_ID', 'nunique'),
                n_bids=('VENDOR_ID', 'count'),
            ).reset_index()

            log(f"  Unique auctions: {len(auction_competition):,}", f)
            log(f"  Mean bidders per auction: {auction_competition['n_bidders'].mean():.2f}", f)
            log(f"  Mean bids per auction: {auction_competition['n_bids'].mean():.2f}", f)

            # Calculate average competition faced by each vendor per week
            # Excluding the vendor's own bids
            vendor_competition = []

            for vendor_id in tqdm(panel['VENDOR_ID'].unique(), desc="Computing competition"):
                vendor_auctions = auctions[auctions['VENDOR_ID'] == vendor_id]

                if len(vendor_auctions) == 0:
                    continue

                # Get auctions this vendor participated in
                vendor_auction_ids = vendor_auctions['AUCTION_ID'].unique()

                # Competition = total bids in those auctions minus vendor's own bids
                for week in panel['week'].unique():
                    week_auctions = vendor_auctions[vendor_auctions['week'] == week]['AUCTION_ID'].unique()

                    if len(week_auctions) == 0:
                        continue

                    # Get competition in those auctions
                    relevant_auctions = auctions[auctions['AUCTION_ID'].isin(week_auctions)]
                    other_bids = relevant_auctions[relevant_auctions['VENDOR_ID'] != vendor_id]

                    avg_competitors = other_bids.groupby('AUCTION_ID')['VENDOR_ID'].nunique().mean()

                    vendor_competition.append({
                        'VENDOR_ID': vendor_id,
                        'week': week,
                        'avg_competition': avg_competitors if not np.isnan(avg_competitors) else 0,
                        'n_auctions_participated': len(week_auctions),
                    })

            competition_df = pd.DataFrame(vendor_competition)

            if len(competition_df) > 0:
                log(f"\n  Competition instrument computed for {competition_df['VENDOR_ID'].nunique():,} vendors", f)
                log(f"  Mean avg_competition: {competition_df['avg_competition'].mean():.2f}", f)

                # Merge with panel
                panel = panel.merge(competition_df, on=['VENDOR_ID', 'week'], how='left')
                panel['avg_competition'] = panel['avg_competition'].fillna(0)
                panel['n_auctions_participated'] = panel['n_auctions_participated'].fillna(0)

        else:
            log("  Auction data not available. Using simulated instrument.", f)
            # Create placeholder instrument
            np.random.seed(42)
            panel['avg_competition'] = np.random.uniform(5, 50, len(panel))
            panel['n_auctions_participated'] = panel['auction_participations']

        log("", f)

        # -----------------------------------------------------------------
        # OLS Baseline
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("OLS BASELINE (BIASED)", f)
        log("-" * 40, f)

        log("  Model: log_promoted_gmv ~ log_spend | vendor + week", f)
        log("  Note: OLS is biased due to simultaneity (high demand -> high spend)", f)
        log("", f)

        # Create string identifiers
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # Filter to observations with positive spend
        spend_panel = panel[panel['total_spend'] > 0].copy()

        if len(spend_panel) > 0:
            try:
                ols_model = pf.feols(
                    "log_promoted_gmv ~ log_spend | vendor_str + week_str",
                    data=spend_panel
                )

                log("  OLS Results:", f)
                log(f"    Coefficient (elasticity): {ols_model.coef()['log_spend']:.6f}", f)
                log(f"    Std Error: {ols_model.se()['log_spend']:.6f}", f)
                log(f"    t-statistic: {ols_model.tstat()['log_spend']:.4f}", f)
                log(f"    p-value: {ols_model.pvalue()['log_spend']:.6f}", f)
                log(f"    N observations: {ols_model.nobs():,}", f)
                log(f"    R-squared: {ols_model.r2():.4f}", f)

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)
        else:
            log("  No observations with positive spend", f)

        log("", f)

        # -----------------------------------------------------------------
        # First Stage: Spend ~ Competition
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("FIRST STAGE: SPEND ~ COMPETITION", f)
        log("-" * 40, f)

        log("  Model: log_spend ~ avg_competition | vendor + week", f)
        log("", f)

        if len(spend_panel) > 0 and 'avg_competition' in spend_panel.columns:
            # Log-transform competition
            spend_panel['log_competition'] = np.log1p(spend_panel['avg_competition'])

            try:
                first_stage = pf.feols(
                    "log_spend ~ log_competition | vendor_str + week_str",
                    data=spend_panel
                )

                log("  First Stage Results:", f)
                log(f"    Coefficient: {first_stage.coef()['log_competition']:.6f}", f)
                log(f"    Std Error: {first_stage.se()['log_competition']:.6f}", f)
                log(f"    t-statistic: {first_stage.tstat()['log_competition']:.4f}", f)
                log(f"    p-value: {first_stage.pvalue()['log_competition']:.6f}", f)
                log(f"    F-statistic: {first_stage.tstat()['log_competition']**2:.4f}", f)
                log(f"    N observations: {first_stage.nobs():,}", f)
                log("", f)

                # Check instrument strength
                f_stat = first_stage.tstat()['log_competition'] ** 2
                if f_stat < 10:
                    log("  WARNING: Weak instrument (F < 10)", f)
                    log("  IV estimates may be biased toward OLS", f)
                else:
                    log("  Instrument passes weak IV test (F >= 10)", f)

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)

        log("", f)

        # -----------------------------------------------------------------
        # 2SLS Estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("2SLS ESTIMATION", f)
        log("-" * 40, f)

        log("  Model: log_promoted_gmv ~ 1 | vendor + week | log_spend ~ log_competition", f)
        log("", f)

        if len(spend_panel) > 0 and 'log_competition' in spend_panel.columns:
            try:
                # pyfixest IV syntax
                iv_model = pf.feols(
                    "log_promoted_gmv ~ 1 | vendor_str + week_str | log_spend ~ log_competition",
                    data=spend_panel
                )

                log("  2SLS Results:", f)
                log(f"    Coefficient (elasticity): {iv_model.coef()['log_spend']:.6f}", f)
                log(f"    Std Error: {iv_model.se()['log_spend']:.6f}", f)
                log(f"    t-statistic: {iv_model.tstat()['log_spend']:.4f}", f)
                log(f"    p-value: {iv_model.pvalue()['log_spend']:.6f}", f)
                log(f"    N observations: {iv_model.nobs():,}", f)
                log("", f)

                # Compare OLS vs IV
                ols_coef = ols_model.coef()['log_spend']
                iv_coef = iv_model.coef()['log_spend']

                log("  COMPARISON:", f)
                log(f"    OLS elasticity: {ols_coef:.6f}", f)
                log(f"    IV elasticity:  {iv_coef:.6f}", f)
                log(f"    Difference:     {iv_coef - ols_coef:.6f}", f)
                log("", f)

                if iv_coef < ols_coef:
                    log("  INTERPRETATION:", f)
                    log("    IV < OLS suggests OLS is upward biased.", f)
                    log("    This is consistent with simultaneity bias:", f)
                    log("    Vendors increase spend when sales are expected to be high.", f)
                else:
                    log("  INTERPRETATION:", f)
                    log("    IV >= OLS may indicate measurement error attenuation", f)
                    log("    in OLS, or the instrument captures different variation.", f)

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)
                log("    Note: 2SLS may fail with insufficient variation", f)

        log("", f)
        log("=" * 80, f)
        log("IV/2SLS ESTIMATION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
