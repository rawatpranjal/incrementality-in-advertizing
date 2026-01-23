#!/usr/bin/env python3
"""
04_main_regressions.py
Estimates primary ad-click → vendor spend models:
1. Model 2: Y_utv = α_u + λ_t + φ_v + β·C_utv + ε
2. Model 2.5: Y_utv = α_ut + φ_v + β·C_utv + ε (User×Week FE)
3. Model 3: Y_stv = α_s + λ_t + φ_v + β·C_stv + ε
4. Two-Part: Conversion (D) + Conditional spend (log Y | Y > 0)

Interpretation: β = dollars of vendor spend per additional sponsored click
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
OUTPUT_FILE = RESULTS_DIR / "04_main_regressions.txt"

SESSION_GAPS = [1, 2, 3, 5, 7]

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("04_MAIN_REGRESSIONS", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script tests the core causal question: does an additional sponsored click", f)
        log("on a vendor cause incremental spend on that vendor? We estimate the coefficient", f)
        log("β in Y ~ C | FE and examine how β changes as we progressively add User, Week,", f)
        log("and Vendor fixed effects. If β remains positive and significant across", f)
        log("specifications, clicks have a causal effect on spend. If β attenuates toward", f)
        log("zero or flips sign with tighter FE, selection bias is present. Model 2.5 uses", f)
        log("User×Week FE to control for weekly purchasing intent, identifying β from", f)
        log("within-(user,week) reallocation across vendors. Model 3 uses Session FE to", f)
        log("absorb browsing intent within a shopping episode. The two-part model separates", f)
        log("the extensive margin (probability of any purchase) from the intensive margin", f)
        log("(conditional spend amount).", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD PANELS
        # ============================================================
        log("Loading panels...", f)

        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        log(f"Panel A (u,t,v): {len(panel_utv):,} rows", f)

        panels_stv = {}
        for gap in SESSION_GAPS:
            panels_stv[gap] = pd.read_parquet(DATA_DIR / f'panel_stv_{gap}d.parquet')
            log(f"Panel B ({gap}d gap): {len(panels_stv[gap]):,} rows", f)

        log("\n--- Panel A Preview ---", f)
        log(str(panel_utv.head()), f)
        log(f"\nColumns: {list(panel_utv.columns)}", f)

        # ============================================================
        # 2. MODEL 2: USER × WEEK × VENDOR
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 2: Y_utv = α_u + λ_t + φ_v + β·C_utv + ε", f)
        log("=" * 80, f)

        df = panel_utv.copy()

        # 2.1 OLS (no FE)
        log("\n--- 2.1 OLS (no fixed effects) ---", f)
        model_ols = pf.feols("Y ~ C", data=df)
        log(str(model_ols.summary()), f)

        # 2.2 User FE only
        log("\n--- 2.2 User FE only ---", f)
        model_user = pf.feols("Y ~ C | user_id", data=df)
        log(str(model_user.summary()), f)

        # 2.3 User + Week FE
        log("\n--- 2.3 User + Week FE ---", f)
        model_user_week = pf.feols("Y ~ C | user_id + year_week", data=df)
        log(str(model_user_week.summary()), f)

        # 2.4 Full Model 2
        log("\n--- 2.4 FULL MODEL 2: User + Week + Vendor FE ---", f)
        model2 = pf.feols("Y ~ C | user_id + year_week + vendor_id", data=df, vcov={'CRV1': 'user_id'})
        log(str(model2.summary()), f)

        # 2.5 Two-way clustering
        log("\n--- 2.5 Model 2 with two-way clustering (user, vendor) ---", f)
        try:
            model2_twoway = pf.feols("Y ~ C | user_id + year_week + vendor_id",
                                     data=df,
                                     vcov={'CRV1': ['user_id', 'vendor_id']})
            log(str(model2_twoway.summary()), f)
        except Exception as e:
            log(f"Two-way clustering error: {e}", f)
            log("Falling back to user clustering", f)

        # 2.6 With controls
        log("\n--- 2.6 Model 2 with auction controls ---", f)
        control_cols = ['avg_rank', 'share_rank1', 'avg_quality', 'avg_pacing']
        available_controls = [c for c in control_cols if c in df.columns and df[c].notna().any()]

        if available_controls:
            for c in available_controls:
                df[c] = df[c].fillna(0)
            formula = f"Y ~ C + {' + '.join(available_controls)} | user_id + year_week + vendor_id"
            log(f"Formula: {formula}", f)
            model2_controls = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
            log(str(model2_controls.summary()), f)
        else:
            log("No controls available", f)

        # 2.7 With impressions
        log("\n--- 2.7 Model 2 with impressions (view-through) ---", f)
        if 'I' in df.columns:
            model2_impressions = pf.feols("Y ~ C + I | user_id + year_week + vendor_id",
                                           data=df, vcov={'CRV1': 'user_id'})
            log(str(model2_impressions.summary()), f)
        else:
            log("Impressions not available", f)

        # ============================================================
        # 2.5 MODEL 2.5: USER×WEEK FE (INTENT-CONTROLLED)
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 2.5: Y_utv = α_ut + φ_v + β·C_utv + ε (User×Week FE)", f)
        log("=" * 80, f)
        log("\nThis model absorbs user-week purchasing intent.", f)
        log("β is identified from within-(user,week) reallocation across vendors.\n", f)

        df = panel_utv.copy()
        df['user_week'] = df['user_id'].astype(str) + '_' + df['year_week'].astype(str)

        n_user_weeks = df['user_week'].nunique()
        vendors_per_uw = df.groupby('user_week')['vendor_id'].nunique()
        multi_vendor_uw = (vendors_per_uw > 1).sum()

        log(f"User-week cells: {n_user_weeks:,}", f)
        log(f"User-weeks with >1 vendor: {multi_vendor_uw:,} ({multi_vendor_uw/n_user_weeks*100:.1f}%)", f)
        log(f"Mean vendors per user-week: {vendors_per_uw.mean():.2f}", f)

        log("\n--- Model 2.5: User×Week FE + Vendor FE ---", f)
        model2_5 = None
        beta_2_5 = None
        try:
            model2_5 = pf.feols("Y ~ C | user_week + vendor_id", data=df, vcov={'CRV1': 'user_id'})
            log(str(model2_5.summary()), f)

            beta_2_5 = model2_5.coef()['C']
            se_2_5 = model2_5.se()['C']

            log(f"\nβ (Model 2.5) = {beta_2_5:.4f} (SE = {se_2_5:.4f})", f)
            log(f"\nComparison with Model 2:", f)
            log(f"  Model 2 (α_u + λ_t + φ_v): β = {model2.coef()['C']:.4f}", f)
            log(f"  Model 2.5 (α_ut + φ_v):   β = {beta_2_5:.4f}", f)

            change = (beta_2_5 - model2.coef()['C']) / abs(model2.coef()['C']) * 100 if model2.coef()['C'] != 0 else 0
            log(f"\n  Change: {change:+.1f}%", f)

        except Exception as e:
            log(f"Error estimating Model 2.5: {e}", f)
            log("User×Week FE may be too fine-grained for this data.", f)

        # ============================================================
        # 3. MODEL 3: SESSION × WEEK × VENDOR
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 3: Y_stv = α_s + λ_t + φ_v + β·C_stv + ε", f)
        log("=" * 80, f)

        results_model3 = []

        for gap_days in tqdm(SESSION_GAPS, desc="Estimating Model 3"):
            df_stv = panels_stv[gap_days].copy()

            log(f"\n--- {gap_days}-day session gap ---", f)
            log(f"Observations: {len(df_stv):,}", f)
            log(f"Sessions: {df_stv['session_id'].nunique():,}", f)

            try:
                model3 = pf.feols("Y ~ C | session_id + year_week + vendor_id",
                                  data=df_stv,
                                  vcov={'CRV1': 'user_id'})

                coef = model3.coef()['C']
                se = model3.se()['C']

                log(f"β = {coef:.4f} (SE = {se:.4f})", f)

                results_model3.append({
                    'gap_days': gap_days,
                    'n_obs': len(df_stv),
                    'n_sessions': df_stv['session_id'].nunique(),
                    'beta': coef,
                    'se': se,
                    't_stat': coef / se
                })
            except Exception as e:
                log(f"Error: {e}", f)

        log("\n--- Model 3 Summary Across Session Gaps ---", f)
        results_df = pd.DataFrame(results_model3)
        log(results_df.to_string(index=False), f)

        # ============================================================
        # 4. TWO-PART MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("TWO-PART MODEL", f)
        log("=" * 80, f)

        df = panel_utv.copy()

        # Part 1: Conversion
        log("\n--- Part 1: Conversion (D = 1{Y > 0}) ---", f)
        log(f"Conversion rate: {df['D'].mean()*100:.2f}%", f)

        model_part1 = pf.feols("D ~ C | user_id + year_week + vendor_id",
                                data=df, vcov={'CRV1': 'user_id'})
        log(str(model_part1.summary()), f)

        beta_D = model_part1.coef()['C']
        log(f"\nβ^D = {beta_D:.6f}", f)
        log(f"Interpretation: 1 additional click → {beta_D*100:.2f} percentage point increase in conversion probability", f)

        # Part 2: Conditional spend
        log("\n--- Part 2: Conditional Spend (log(1+Y) | Y > 0) ---", f)
        df_converters = df[df['D'] == 1].copy()
        log(f"Converters: {len(df_converters):,} ({len(df_converters)/len(df)*100:.1f}%)", f)

        beta_Y = 0
        try:
            model_part2 = pf.feols("log_Y ~ C | user_id + year_week + vendor_id",
                                    data=df_converters, vcov={'CRV1': 'user_id'})
            log(str(model_part2.summary()), f)
            beta_Y = model_part2.coef()['C']
            log(f"\nβ^Y = {beta_Y:.4f}", f)
            log(f"Interpretation: 1 additional click → {(np.exp(beta_Y)-1)*100:.1f}% increase in conditional spend", f)
        except Exception as e:
            log(f"Error estimating Part 2: {e}", f)
            log("Insufficient variation among converters for full FE model", f)
            # Try simpler model
            try:
                model_part2 = pf.feols("log_Y ~ C | user_id", data=df_converters)
                beta_Y = model_part2.coef()['C']
                log(f"\nβ^Y (user FE only) = {beta_Y:.4f}", f)
            except:
                log("Cannot estimate Part 2 - too few converters", f)

        # ============================================================
        # 5. VENDOR-SPECIFIC ROI
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("VENDOR-SPECIFIC ROI", f)
        log("=" * 80, f)

        vendor_clicks = panel_utv.groupby('vendor_id')['C'].sum().sort_values(ascending=False)
        top_vendors = vendor_clicks.head(20).index.tolist()

        log(f"Estimating β for top {len(top_vendors)} vendors by click volume...", f)

        vendor_results = []
        for vendor in tqdm(top_vendors, desc="Vendors"):
            df_vendor = panel_utv[panel_utv['vendor_id'] == vendor].copy()

            if len(df_vendor) < 100:
                continue

            try:
                model = pf.feols("Y ~ C | user_id + year_week", data=df_vendor)

                vendor_results.append({
                    'vendor_id': vendor[:20],
                    'n_obs': len(df_vendor),
                    'total_clicks': df_vendor['C'].sum(),
                    'total_spend': df_vendor['Y'].sum(),
                    'beta': model.coef()['C'],
                    'se': model.se()['C']
                })
            except:
                pass

        vendor_df = pd.DataFrame(vendor_results)
        log("\n--- Vendor-Specific β Estimates ---", f)
        log(vendor_df.to_string(index=False), f)

        # ============================================================
        # 6. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MAIN REGRESSION RESULTS SUMMARY", f)
        log("=" * 80, f)

        log("\n--- Model 2: Y_utv = α_u + λ_t + φ_v + β·C_utv + ε ---", f)
        log(f"β = {model2.coef()['C']:.4f} (SE = {model2.se()['C']:.4f})", f)
        log(f"Interpretation: 1 additional sponsored click → ${model2.coef()['C']:.2f} additional vendor spend", f)

        log("\n--- Model 2.5: Y_utv = α_ut + φ_v + β·C_utv + ε (Intent-Controlled) ---", f)
        if model2_5 is not None:
            log(f"β = {model2_5.coef()['C']:.4f} (SE = {model2_5.se()['C']:.4f})", f)
            log("(User×Week FE absorbs weekly purchasing intent)", f)
        else:
            log("Model 2.5 not estimated (insufficient within-user-week variation)", f)

        log("\n--- Model 3: Y_stv = α_s + λ_t + φ_v + β·C_stv + ε ---", f)
        log("(Session FE absorbs browsing intent)", f)
        for r in results_model3:
            log(f"  {r['gap_days']}d gap: β = {r['beta']:.4f} (SE = {r['se']:.4f})", f)

        log("\n--- Two-Part Model ---", f)
        log(f"Part 1 (Conversion): β^D = {beta_D:.6f}", f)
        log(f"Part 2 (Cond. Spend): β^Y = {beta_Y:.4f}", f)

        # Save results to JSON
        results_summary = {
            'model2_beta': model2.coef()['C'],
            'model2_se': model2.se()['C'],
            'model2_5_beta': model2_5.coef()['C'] if model2_5 is not None else None,
            'model2_5_se': model2_5.se()['C'] if model2_5 is not None else None,
            'model3_results': results_model3,
            'twopart_beta_D': beta_D,
            'twopart_beta_Y': beta_Y,
            'vendor_results': vendor_results
        }

        with open(DATA_DIR / 'regression_results.json', 'w') as jf:
            json.dump(results_summary, jf, indent=2, default=str)

        log(f"\nResults saved to {DATA_DIR / 'regression_results.json'}", f)

if __name__ == "__main__":
    main()
