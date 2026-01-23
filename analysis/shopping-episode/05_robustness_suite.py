#!/usr/bin/env python3
"""
05_robustness_suite.py
Validates main regression results through multiple robustness checks:
1. Promoted-Halo Lower Bound: Clicked-item vs other-vendor-items spend
2. Delayed Conversion: Window sweep (L = 0, 1, 2, 4 weeks)
3. Position Bias: Auction controls + rank stratification
4. View-Through: Impressions effect
5. Placebo Test: Past spend on future clicks
6. New-to-Vendor: First-click subsample
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "05_robustness_suite.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("05_ROBUSTNESS_SUITE", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script tests whether the main result is robust to alternative specifications.", f)
        log("The placebo test regresses past spend on future clicks to detect reverse causality", f)
        log("(users who spend more become more likely to click later). The rank heterogeneity", f)
        log("test examines whether clicks in top auction positions (rank 1-3) have different", f)
        log("effects than lower-ranked clicks. The window sweep tests whether clicks cause", f)
        log("delayed purchases by accumulating spend over L weeks post-click. The conversion", f)
        log("model tests whether clicks affect the probability of any purchase (extensive", f)
        log("margin) separately from the amount spent (intensive margin). The halo test", f)
        log("separates spend on clicked items from spend on other items from the same vendor,", f)
        log("providing a lower bound on vendor-level spillover effects.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("Loading data...", f)

        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        purchases_mapped = pd.read_parquet(DATA_DIR / 'purchases_mapped.parquet')

        log(f"Panel (u,t,v): {len(panel_utv):,} rows", f)
        log(f"Promoted events: {len(promoted_events):,} rows", f)
        log(f"Purchases mapped: {len(purchases_mapped):,} rows", f)

        # ============================================================
        # 2. PROMOTED-HALO LOWER BOUND
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("2. PROMOTED-HALO LOWER BOUND", f)
        log("=" * 80, f)
        log("\nCAVEAT: This is a LOWER BOUND - organic vendor purchases are unobserved.", f)
        log("Only purchases linkable to promoted events can be attributed to vendor.\n", f)

        promoted_events['click_time'] = pd.to_datetime(promoted_events['click_time'])
        promoted_events['year_week'] = promoted_events['click_time'].dt.isocalendar().week.astype(str).str.zfill(2)
        promoted_events['year'] = promoted_events['click_time'].dt.year
        promoted_events['year_week'] = promoted_events['year'].astype(str) + '_W' + promoted_events['year_week']

        clicked_products = promoted_events[['user_id', 'vendor_id', 'product_id']].drop_duplicates()
        clicked_products['clicked'] = True

        log(f"Unique (user, vendor, product) clicked: {len(clicked_products):,}", f)

        purchases_valid = purchases_mapped[purchases_mapped['is_post_click']].copy()
        purchases_valid['purchase_time'] = pd.to_datetime(purchases_valid['purchase_time'])
        purchases_valid['year_week'] = purchases_valid['purchase_time'].dt.isocalendar().week.astype(str).str.zfill(2)
        purchases_valid['year'] = purchases_valid['purchase_time'].dt.year
        purchases_valid['year_week'] = purchases_valid['year'].astype(str) + '_W' + purchases_valid['year_week']

        purchases_valid = purchases_valid.merge(
            clicked_products,
            left_on=['user_id', 'click_vendor_id', 'product_id'],
            right_on=['user_id', 'vendor_id', 'product_id'],
            how='left'
        )
        purchases_valid['clicked'] = purchases_valid['clicked'].fillna(False)

        spend_clicked = purchases_valid[purchases_valid['clicked']].groupby(
            ['user_id', 'year_week', 'click_vendor_id']
        )['spend'].sum().reset_index()
        spend_clicked.columns = ['user_id', 'year_week', 'vendor_id', 'Y_clicked']

        spend_other = purchases_valid[~purchases_valid['clicked']].groupby(
            ['user_id', 'year_week', 'click_vendor_id']
        )['spend'].sum().reset_index()
        spend_other.columns = ['user_id', 'year_week', 'vendor_id', 'Y_other']

        log(f"Spend on clicked items: ${spend_clicked['Y_clicked'].sum():,.2f}", f)
        log(f"Spend on other items: ${spend_other['Y_other'].sum():,.2f}", f)

        panel_halo = panel_utv.merge(spend_clicked, on=['user_id', 'year_week', 'vendor_id'], how='left')
        panel_halo = panel_halo.merge(spend_other, on=['user_id', 'year_week', 'vendor_id'], how='left')
        panel_halo['Y_clicked'] = panel_halo['Y_clicked'].fillna(0)
        panel_halo['Y_other'] = panel_halo['Y_other'].fillna(0)

        log("\n--- Effect on Clicked-Item Spend ---", f)
        model_clicked = pf.feols("Y_clicked ~ C | user_id + year_week + vendor_id",
                                  data=panel_halo, vcov={'CRV1': 'user_id'})
        log(f"β (clicked items) = {model_clicked.coef()['C']:.4f} (SE = {model_clicked.se()['C']:.4f})", f)

        log("\n--- Effect on Other-Item Spend (Promoted-Halo Lower Bound) ---", f)
        model_other = pf.feols("Y_other ~ C | user_id + year_week + vendor_id",
                                data=panel_halo, vcov={'CRV1': 'user_id'})
        log(f"β (other items) = {model_other.coef()['C']:.4f} (SE = {model_other.se()['C']:.4f})", f)

        log("\n*** INTERPRETATION ***", f)
        log("β_other > 0 indicates promoted-halo effect (click leads to other vendor purchases)", f)
        log("This is a LOWER BOUND: organic vendor purchases after ad exposure are unobserved.", f)

        # ============================================================
        # 3. DELAYED CONVERSION (WINDOW SWEEP)
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("3. DELAYED CONVERSION (WINDOW SWEEP)", f)
        log("=" * 80, f)

        weeks_ordered = sorted(panel_utv['year_week'].unique())
        week_to_idx = {w: i for i, w in enumerate(weeks_ordered)}
        panel_utv['week_idx'] = panel_utv['year_week'].map(week_to_idx)

        log(f"Weeks available: {len(weeks_ordered)}", f)
        log(f"Range: {weeks_ordered[0]} to {weeks_ordered[-1]}", f)

        WINDOWS = [0, 1, 2, 4]
        window_results = []

        for L in tqdm(WINDOWS, desc="Window sweep"):
            panel_window = panel_utv.copy()
            spend_by_week = panel_utv.groupby(['user_id', 'vendor_id', 'week_idx'])['Y'].sum().reset_index()

            for week_offset in range(L + 1):
                spend_offset = spend_by_week.copy()
                spend_offset['week_idx'] = spend_offset['week_idx'] - week_offset
                spend_offset = spend_offset.rename(columns={'Y': f'Y_offset_{week_offset}'})

                panel_window = panel_window.merge(
                    spend_offset[['user_id', 'vendor_id', 'week_idx', f'Y_offset_{week_offset}']],
                    on=['user_id', 'vendor_id', 'week_idx'],
                    how='left'
                )
                panel_window[f'Y_offset_{week_offset}'] = panel_window[f'Y_offset_{week_offset}'].fillna(0)

            offset_cols = [f'Y_offset_{i}' for i in range(L + 1)]
            panel_window[f'Y_L{L}'] = panel_window[offset_cols].sum(axis=1)

            try:
                model = pf.feols(f"Y_L{L} ~ C | user_id + year_week + vendor_id",
                                 data=panel_window, vcov={'CRV1': 'user_id'})
                beta = model.coef()['C']
                se = model.se()['C']

                window_results.append({
                    'L_weeks': L,
                    'beta': beta,
                    'se': se,
                    'mean_Y': panel_window[f'Y_L{L}'].mean()
                })

                log(f"L={L} weeks: β = {beta:.4f} (SE = {se:.4f}), mean Y^(L) = ${panel_window[f'Y_L{L}'].mean():.2f}", f)
            except Exception as e:
                log(f"L={L} weeks: Error - {e}", f)

        log("\n--- Window Sweep Summary ---", f)
        window_df = pd.DataFrame(window_results)
        log(window_df.to_string(index=False), f)
        log("\nInterpretation: β increasing with L indicates delayed conversion effect", f)

        # ============================================================
        # 4. POSITION BIAS (RANK STRATIFICATION)
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("4. POSITION BIAS (RANK STRATIFICATION)", f)
        log("=" * 80, f)
        log("\nCAVEAT: Auction controls are POST-TREATMENT/ENDOGENOUS.", f)
        log("Coefficients are ASSOCIATIONS, not causal effects.\n", f)

        df = panel_utv.copy()

        control_cols = ['avg_rank', 'share_rank1', 'avg_quality', 'avg_pacing', 'avg_final_bid']
        for col in control_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        log("\n--- Decomposition Model with Auction Controls (NOT CAUSAL) ---", f)
        available_controls = [c for c in control_cols if c in df.columns]

        if available_controls:
            formula = f"Y ~ C + {' + '.join(available_controls)} | user_id + year_week + vendor_id"
            log(f"Formula: {formula}", f)
            model_controls = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
            log(str(model_controls.summary()), f)

        log("\n--- Stratification by Rank ---", f)
        model_top = None
        model_low = None

        if 'share_rank1' in df.columns:
            df_top_rank = df[df['share_rank1'] > 0.5].copy()
            df_low_rank = df[(df['share_rank1'] <= 0.5) & (df['share_rank1'] > 0)].copy()

            log(f"Top-rank subsample (share_rank1 > 0.5): {len(df_top_rank):,} obs", f)
            log(f"Lower-rank subsample (0 < share_rank1 <= 0.5): {len(df_low_rank):,} obs", f)

            if len(df_top_rank) > 100:
                try:
                    model_top = pf.feols("Y ~ C | user_id + year_week + vendor_id",
                                          data=df_top_rank, vcov={'CRV1': 'user_id'})
                    log(f"\nTop-rank β = {model_top.coef()['C']:.4f} (SE = {model_top.se()['C']:.4f})", f)
                except Exception as e:
                    log(f"\nTop-rank model error: {e}", f)

            if len(df_low_rank) > 100:
                try:
                    model_low = pf.feols("Y ~ C | user_id + year_week + vendor_id",
                                          data=df_low_rank, vcov={'CRV1': 'user_id'})
                    log(f"Lower-rank β = {model_low.coef()['C']:.4f} (SE = {model_low.se()['C']:.4f})", f)
                except Exception as e:
                    log(f"Lower-rank model error: {e}", f)

        # ============================================================
        # 5. VIEW-THROUGH EFFECT
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("5. VIEW-THROUGH EFFECT (IMPRESSIONS)", f)
        log("=" * 80, f)

        df = panel_utv.copy()
        model_viewthrough = None

        if 'I' in df.columns:
            log(f"\nImpressions available: mean = {df['I'].mean():.1f}, max = {df['I'].max()}", f)

            model_viewthrough = pf.feols("Y ~ C + I | user_id + year_week + vendor_id",
                                          data=df, vcov={'CRV1': 'user_id'})
            log("\n--- Model with Clicks + Impressions ---", f)
            log(str(model_viewthrough.summary()), f)

            beta_C = model_viewthrough.coef()['C']
            beta_I = model_viewthrough.coef()['I']
            log(f"\nβ_C (click effect) = {beta_C:.4f}", f)
            log(f"β_I (view-through) = {beta_I:.4f}", f)
        else:
            log("Impressions not available in panel", f)

        # ============================================================
        # 6. PLACEBO TEST
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("6. PLACEBO TEST (Past Spend ~ Future Clicks)", f)
        log("=" * 80, f)

        df = panel_utv.copy()
        df = df.sort_values(['user_id', 'vendor_id', 'week_idx'])
        df['Y_lag1'] = df.groupby(['user_id', 'vendor_id'])['Y'].shift(1)
        df_placebo = df[df['Y_lag1'].notna()].copy()

        log(f"Placebo sample: {len(df_placebo):,} observations", f)

        log("\n--- Placebo: Y_{t-1} ~ C_t ---", f)
        model_placebo = pf.feols("Y_lag1 ~ C | user_id + year_week + vendor_id",
                                  data=df_placebo, vcov={'CRV1': 'user_id'})
        log(str(model_placebo.summary()), f)

        beta_placebo = model_placebo.coef()['C']
        se_placebo = model_placebo.se()['C']

        log(f"\nβ^pl = {beta_placebo:.4f} (SE = {se_placebo:.4f})", f)
        log(f"t-stat = {beta_placebo / se_placebo:.2f}", f)

        if abs(beta_placebo / se_placebo) < 1.96:
            log("\n✓ Placebo test PASSED: No significant relationship between past spend and future clicks", f)
        else:
            log("\n⚠ Placebo test WARNING: Significant relationship detected - possible selection/anticipation", f)

        # ============================================================
        # 7. NEW-TO-VENDOR SUBSAMPLE
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("7. NEW-TO-VENDOR SUBSAMPLE", f)
        log("=" * 80, f)

        first_click = panel_utv[panel_utv['C'] > 0].groupby(['user_id', 'vendor_id'])['week_idx'].min().reset_index()
        first_click.columns = ['user_id', 'vendor_id', 'first_click_week']

        df = panel_utv.merge(first_click, on=['user_id', 'vendor_id'], how='left')
        df_new = df[df['week_idx'] == df['first_click_week']].copy()

        log(f"New-to-vendor observations: {len(df_new):,}", f)
        log(f"Share of total: {len(df_new)/len(panel_utv)*100:.1f}%", f)

        model_new = None
        log("\n--- Model on New-to-Vendor Subsample ---", f)
        if len(df_new) > 100:
            try:
                model_new = pf.feols("Y ~ C | user_id + year_week + vendor_id",
                                      data=df_new, vcov={'CRV1': 'user_id'})
                log(str(model_new.summary()), f)
                log(f"\nβ (new-to-vendor) = {model_new.coef()['C']:.4f}", f)
                log("Interpretation: Effect for users with no prior vendor relationship", f)
            except Exception as e:
                log(f"Error estimating new-to-vendor model: {e}", f)
        else:
            log("Insufficient observations for new-to-vendor analysis", f)

        # ============================================================
        # 8. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ROBUSTNESS SUITE SUMMARY", f)
        log("=" * 80, f)

        log("\n1. PROMOTED-HALO LOWER BOUND", f)
        log("   (Organic vendor purchases are unobserved - true halo may be higher)", f)
        log(f"   β (clicked items): {model_clicked.coef()['C']:.4f}", f)
        log(f"   β (other items):   {model_other.coef()['C']:.4f}", f)

        log("\n2. WINDOW SWEEP", f)
        for r in window_results:
            log(f"   L={r['L_weeks']} weeks: β = {r['beta']:.4f}", f)

        log("\n3. POSITION BIAS (Decomposition Only - NOT Causal)", f)
        if model_top is not None:
            log(f"   Top-rank β: {model_top.coef()['C']:.4f}", f)
        if model_low is not None:
            log(f"   Lower-rank β: {model_low.coef()['C']:.4f}", f)

        log("\n4. VIEW-THROUGH", f)
        if model_viewthrough is not None:
            log(f"   β_C (clicks): {model_viewthrough.coef()['C']:.4f}", f)
            log(f"   β_I (impressions): {model_viewthrough.coef()['I']:.4f}", f)

        log("\n5. PLACEBO TEST", f)
        log(f"   β^pl = {beta_placebo:.4f} (t = {beta_placebo/se_placebo:.2f})", f)
        log(f"   Status: {'PASSED' if abs(beta_placebo/se_placebo) < 1.96 else 'WARNING'}", f)

        log("\n6. NEW-TO-VENDOR", f)
        if model_new is not None:
            log(f"   β = {model_new.coef()['C']:.4f}", f)

        # Save results
        robustness_results = {
            'promoted_halo_lower_bound': {
                'beta_clicked': model_clicked.coef()['C'],
                'beta_other': model_other.coef()['C'],
                'caveat': 'LOWER BOUND - organic vendor purchases are unobserved'
            },
            'window_sweep': window_results,
            'placebo': {
                'beta': beta_placebo,
                'se': se_placebo,
                'passed': abs(beta_placebo/se_placebo) < 1.96
            },
            'position_bias_caveat': 'Auction controls are post-treatment/endogenous - associations only'
        }

        with open(DATA_DIR / 'robustness_results.json', 'w') as jf:
            json.dump(robustness_results, jf, indent=2, default=str)

        log(f"\nResults saved to {DATA_DIR / 'robustness_results.json'}", f)

        log("\n" + "=" * 80, f)
        log("ROBUSTNESS SUITE COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
