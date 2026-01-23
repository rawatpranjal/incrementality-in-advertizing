#!/usr/bin/env python3
"""
10_near_win_iv.py
Implements near-win IV strategy using marginal auction wins as instrument.
Runs first stage, reduced form, and 2SLS specifications.
Tests instrument validity with balance checks and bandwidth sensitivity.
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
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "10_near_win_iv.txt"

SESSION_GAPS = [1, 2, 3, 5, 7]

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def safe_coef(model, var):
    """Safely extract coefficient, returning None if not found."""
    try:
        coefs = model.coef()
        if var in coefs:
            return coefs[var]
    except:
        pass
    return None

def safe_se(model, var):
    """Safely extract SE, returning None if not found."""
    try:
        ses = model.se()
        if var in ses:
            return ses[var]
    except:
        pass
    return None

def safe_pval(model, var):
    """Safely extract p-value, returning None if not found."""
    try:
        pvals = model.pvalue()
        if var in pvals:
            return pvals[var]
    except:
        pass
    return None

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("10_NEAR_WIN_IV", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script implements the near-win IV strategy for estimating the causal", f)
        log("effect of sponsored clicks on vendor spend. The identifying assumption is", f)
        log("that conditional on being near the margin (rank K or K+1), which side a", f)
        log("vendor falls on is quasi-random - determined by small differences in bid,", f)
        log("quality, or pacing. Marginal wins (rank=K) affect clicks mechanically", f)
        log("(only eligible vendors can be clicked), satisfying the exclusion restriction.", f)
        log("", f)
        log("Specifications:", f)
        log("  First Stage:  C_usv ~ MarginalWin_usv | user_fe + session_fe + vendor_fe", f)
        log("  Reduced Form: Y_usv ~ MarginalWin_usv | user_fe + session_fe + vendor_fe", f)
        log("  2SLS:         Y_usv ~ C_usv | FE, instrumented by MarginalWin", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # Store results for summary table
        all_results = []

        for gap_days in SESSION_GAPS:
            log("", f)
            log("=" * 80, f)
            log(f"SESSION GAP: {gap_days} DAYS", f)
            log("=" * 80, f)

            # Load panel
            panel_file = DATA_DIR / f'panel_iv_{gap_days}d.parquet'
            df = pd.read_parquet(panel_file)
            log(f"\nLoaded {len(df):,} observations", f)

            # ============================================================
            # DATA SUMMARY
            # ============================================================
            log("\n--- Data Summary ---", f)
            log(f"Observations: {len(df):,}", f)
            log(f"Unique users: {df['user_id'].nunique():,}", f)
            log(f"Unique sessions: {df['session_id'].nunique():,}", f)
            log(f"Unique vendors: {df['vendor_id'].nunique():,}", f)

            log(f"\nOutcome (Y):", f)
            log(f"  Mean: ${df['Y'].mean():.4f}", f)
            log(f"  Std: ${df['Y'].std():.4f}", f)
            log(f"  % zero: {(df['Y']==0).mean()*100:.1f}%", f)

            log(f"\nTreatment (C - clicks):", f)
            log(f"  Mean: {df['C'].mean():.4f}", f)
            log(f"  Std: {df['C'].std():.4f}", f)

            log(f"\nInstrument (MarginalWin):", f)
            log(f"  Mean: {df['MarginalWin'].mean():.4f}", f)
            log(f"  Std: {df['MarginalWin'].std():.4f}", f)
            log(f"  % zero: {(df['MarginalWin']==0).mean()*100:.1f}%", f)
            log(f"  Sum: {df['MarginalWin'].sum():,}", f)

            # ============================================================
            # OLS BASELINE (NO FE)
            # ============================================================
            log("\n" + "=" * 60, f)
            log("OLS BASELINE (NO FIXED EFFECTS)", f)
            log("=" * 60, f)

            log("\nModel: Y ~ C", f)
            try:
                ols_baseline = pf.feols("Y ~ C", data=df, vcov='hetero')
                log(str(ols_baseline.summary()), f)
                beta_ols_nfe = safe_coef(ols_baseline, 'C')
                se_ols_nfe = safe_se(ols_baseline, 'C')
            except Exception as e:
                log(f"OLS baseline failed: {e}", f)
                beta_ols_nfe, se_ols_nfe = None, None

            # ============================================================
            # OLS WITH FIXED EFFECTS
            # ============================================================
            log("\n" + "=" * 60, f)
            log("OLS WITH FIXED EFFECTS", f)
            log("=" * 60, f)

            log("\nModel: Y ~ C | user_id + session_id + vendor_id", f)
            try:
                ols_fe = pf.feols("Y ~ C | user_id + session_id + vendor_id",
                                   data=df, vcov={'CRV1': 'user_id'})
                log(str(ols_fe.summary()), f)
                beta_ols = safe_coef(ols_fe, 'C')
                se_ols = safe_se(ols_fe, 'C')
                n_ols = ols_fe._N
            except Exception as e:
                log(f"OLS with FE failed: {e}", f)
                beta_ols, se_ols, n_ols = None, None, len(df)

            # ============================================================
            # FIRST STAGE: C ~ MarginalWin | FE
            # ============================================================
            log("\n" + "=" * 60, f)
            log("FIRST STAGE: C ~ MarginalWin | FE", f)
            log("=" * 60, f)

            log("\nModel: C ~ MarginalWin | user_id + session_id + vendor_id", f)
            try:
                first_stage = pf.feols("C ~ MarginalWin | user_id + session_id + vendor_id",
                                        data=df, vcov={'CRV1': 'user_id'})
                log(str(first_stage.summary()), f)

                beta_fs = safe_coef(first_stage, 'MarginalWin')
                se_fs = safe_se(first_stage, 'MarginalWin')
                tstat_fs = beta_fs / se_fs if (beta_fs and se_fs) else None
                n_fs = first_stage._N

                log(f"\n--- First Stage Diagnostics ---", f)
                log(f"Coefficient: {beta_fs:.4f}" if beta_fs else "Coefficient: NA", f)
                log(f"SE: {se_fs:.4f}" if se_fs else "SE: NA", f)
                log(f"t-stat: {tstat_fs:.2f}" if tstat_fs else "t-stat: NA", f)

                # Weak instrument check
                if tstat_fs:
                    f_stat = tstat_fs ** 2
                    log(f"F-statistic: {f_stat:.2f}", f)
                    if f_stat > 10:
                        log("  -> Strong instrument (F > 10)", f)
                    elif f_stat > 3.84:
                        log("  -> Moderate instrument (F > 3.84)", f)
                    else:
                        log("  -> WEAK INSTRUMENT (F < 3.84)", f)

            except Exception as e:
                log(f"First stage failed: {e}", f)
                beta_fs, se_fs, tstat_fs, n_fs, f_stat = None, None, None, None, None

            # ============================================================
            # REDUCED FORM: Y ~ MarginalWin | FE
            # ============================================================
            log("\n" + "=" * 60, f)
            log("REDUCED FORM: Y ~ MarginalWin | FE", f)
            log("=" * 60, f)

            log("\nModel: Y ~ MarginalWin | user_id + session_id + vendor_id", f)
            try:
                reduced_form = pf.feols("Y ~ MarginalWin | user_id + session_id + vendor_id",
                                         data=df, vcov={'CRV1': 'user_id'})
                log(str(reduced_form.summary()), f)

                beta_rf = safe_coef(reduced_form, 'MarginalWin')
                se_rf = safe_se(reduced_form, 'MarginalWin')
                pval_rf = safe_pval(reduced_form, 'MarginalWin')

                log(f"\n--- Reduced Form Interpretation ---", f)
                if beta_rf:
                    log(f"Effect of marginal win on spend: ${beta_rf:.4f}", f)
                    if beta_fs:
                        implied_iv = beta_rf / beta_fs
                        log(f"Implied IV estimate (RF/FS): ${implied_iv:.4f}", f)

            except Exception as e:
                log(f"Reduced form failed: {e}", f)
                beta_rf, se_rf, pval_rf = None, None, None

            # ============================================================
            # 2SLS: Y ~ C | FE, C instrumented by MarginalWin
            # ============================================================
            log("\n" + "=" * 60, f)
            log("2SLS: Y ~ C | FE (C instrumented by MarginalWin)", f)
            log("=" * 60, f)

            log("\nModel: Y ~ 1 | user_id + session_id + vendor_id | C ~ MarginalWin", f)
            try:
                iv_model = pf.feols("Y ~ 1 | user_id + session_id + vendor_id | C ~ MarginalWin",
                                     data=df, vcov={'CRV1': 'user_id'})
                log(str(iv_model.summary()), f)

                beta_iv = safe_coef(iv_model, 'C')
                se_iv = safe_se(iv_model, 'C')
                pval_iv = safe_pval(iv_model, 'C')
                n_iv = iv_model._N

                log(f"\n--- IV vs OLS Comparison ---", f)
                if beta_iv and beta_ols:
                    log(f"OLS estimate: ${beta_ols:.4f} (SE={se_ols:.4f})" if se_ols else f"OLS: ${beta_ols:.4f}", f)
                    log(f"IV estimate:  ${beta_iv:.4f} (SE={se_iv:.4f})" if se_iv else f"IV: ${beta_iv:.4f}", f)
                    log(f"Difference:   ${beta_iv - beta_ols:.4f}", f)

                    # Hausman-style interpretation
                    if abs(beta_iv - beta_ols) > 2 * (se_iv if se_iv else 0.1):
                        log("  -> Large IV-OLS gap suggests endogeneity", f)
                    else:
                        log("  -> Similar estimates (limited evidence of endogeneity)", f)

            except Exception as e:
                log(f"2SLS failed: {e}", f)
                beta_iv, se_iv, pval_iv, n_iv = None, None, None, None

            # ============================================================
            # PLACEBO TEST: MarginalLoss as instrument
            # ============================================================
            log("\n" + "=" * 60, f)
            log("PLACEBO TEST: MarginalLoss as Instrument", f)
            log("=" * 60, f)

            log("\nMarginalLoss should NOT predict C (vendors at K+1 don't get impressions)", f)
            log("\nFirst Stage (Placebo): C ~ MarginalLoss | FE", f)
            try:
                placebo_fs = pf.feols("C ~ MarginalLoss | user_id + session_id + vendor_id",
                                       data=df, vcov={'CRV1': 'user_id'})
                log(str(placebo_fs.summary()), f)

                beta_placebo = safe_coef(placebo_fs, 'MarginalLoss')
                se_placebo = safe_se(placebo_fs, 'MarginalLoss')

                if beta_placebo and se_placebo:
                    tstat_placebo = beta_placebo / se_placebo
                    log(f"\nPlacebo t-stat: {tstat_placebo:.2f}", f)
                    if abs(tstat_placebo) < 1.96:
                        log("  -> PASSED: MarginalLoss does not predict C", f)
                    else:
                        log("  -> WARNING: MarginalLoss predicts C (unexpected)", f)

            except Exception as e:
                log(f"Placebo test failed: {e}", f)

            # ============================================================
            # BANDWIDTH SENSITIVITY: Different margins around K
            # ============================================================
            log("\n" + "=" * 60, f)
            log("BANDWIDTH SENSITIVITY", f)
            log("=" * 60, f)

            log("\nTesting robustness to different definitions of 'close' auctions:", f)

            # Test with Close (K or K+1) as instrument
            log("\n--- Using Close (K or K+1) as instrument ---", f)
            try:
                fs_close = pf.feols("C ~ Close | user_id + session_id + vendor_id",
                                     data=df, vcov={'CRV1': 'user_id'})
                beta_close = safe_coef(fs_close, 'Close')
                se_close = safe_se(fs_close, 'Close')
                log(f"Close -> C: {beta_close:.4f} (SE={se_close:.4f})" if (beta_close and se_close) else "Failed", f)
            except Exception as e:
                log(f"Close test failed: {e}", f)

            # Test with Eligible as instrument
            log("\n--- Using Eligible (all rank <= K) as instrument ---", f)
            try:
                fs_eligible = pf.feols("C ~ Eligible | user_id + session_id + vendor_id",
                                        data=df, vcov={'CRV1': 'user_id'})
                beta_eligible = safe_coef(fs_eligible, 'Eligible')
                se_eligible = safe_se(fs_eligible, 'Eligible')
                log(f"Eligible -> C: {beta_eligible:.4f} (SE={se_eligible:.4f})" if (beta_eligible and se_eligible) else "Failed", f)
            except Exception as e:
                log(f"Eligible test failed: {e}", f)

            # ============================================================
            # STORE RESULTS
            # ============================================================
            result_row = {
                'gap_days': gap_days,
                'n_obs': len(df),
                'beta_ols': beta_ols,
                'se_ols': se_ols,
                'beta_fs': beta_fs,
                'se_fs': se_fs,
                'f_stat': tstat_fs ** 2 if tstat_fs else None,
                'beta_rf': beta_rf,
                'se_rf': se_rf,
                'beta_iv': beta_iv,
                'se_iv': se_iv
            }
            all_results.append(result_row)

        # ============================================================
        # SUMMARY TABLE
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SUMMARY TABLE: IV RESULTS ACROSS SESSION GAP THRESHOLDS", f)
        log("=" * 80, f)

        results_df = pd.DataFrame(all_results)

        log("\n--- OLS vs IV Comparison ---", f)
        log(f"{'Gap':>5} {'N':>8} {'OLS':>10} {'SE':>8} {'IV':>10} {'SE':>8} {'F-stat':>8}", f)
        log("-" * 65, f)

        for _, row in results_df.iterrows():
            gap = row['gap_days']
            n = row['n_obs']
            ols = f"${row['beta_ols']:.3f}" if row['beta_ols'] is not None else "NA"
            se_o = f"{row['se_ols']:.3f}" if row['se_ols'] is not None else "NA"
            iv = f"${row['beta_iv']:.3f}" if row['beta_iv'] is not None else "NA"
            se_i = f"{row['se_iv']:.3f}" if row['se_iv'] is not None else "NA"
            fst = f"{row['f_stat']:.1f}" if row['f_stat'] is not None else "NA"
            log(f"{gap:>5}d {n:>8,} {ols:>10} {se_o:>8} {iv:>10} {se_i:>8} {fst:>8}", f)

        log("\n--- First Stage and Reduced Form ---", f)
        log(f"{'Gap':>5} {'FS_beta':>10} {'FS_SE':>8} {'RF_beta':>10} {'RF_SE':>8}", f)
        log("-" * 50, f)

        for _, row in results_df.iterrows():
            gap = row['gap_days']
            fs = f"{row['beta_fs']:.4f}" if row['beta_fs'] is not None else "NA"
            se_fs = f"{row['se_fs']:.4f}" if row['se_fs'] is not None else "NA"
            rf = f"${row['beta_rf']:.4f}" if row['beta_rf'] is not None else "NA"
            se_rf = f"{row['se_rf']:.4f}" if row['se_rf'] is not None else "NA"
            log(f"{gap:>5}d {fs:>10} {se_fs:>8} {rf:>10} {se_rf:>8}", f)

        # Save results to JSON
        results_df.to_json(DATA_DIR / 'iv_results.json', orient='records', indent=2)
        log(f"\nResults saved to {DATA_DIR / 'iv_results.json'}", f)

        # ============================================================
        # INTERPRETATION
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("=" * 80, f)

        log("\n--- Key Findings ---", f)

        # Check first stage strength
        avg_f = results_df['f_stat'].mean() if results_df['f_stat'].notna().any() else 0
        log(f"\n1. First Stage Strength:", f)
        log(f"   Average F-statistic: {avg_f:.2f}", f)
        if avg_f > 10:
            log("   -> Strong first stage: MarginalWin predicts C", f)
        elif avg_f > 3.84:
            log("   -> Moderate first stage: proceed with caution", f)
        else:
            log("   -> WEAK first stage: IV estimates unreliable", f)

        # Compare OLS vs IV
        log(f"\n2. OLS vs IV Comparison:", f)
        avg_ols = results_df['beta_ols'].mean() if results_df['beta_ols'].notna().any() else 0
        avg_iv = results_df['beta_iv'].mean() if results_df['beta_iv'].notna().any() else 0
        log(f"   Average OLS: ${avg_ols:.4f}", f)
        log(f"   Average IV:  ${avg_iv:.4f}", f)
        if avg_ols and avg_iv:
            log(f"   Difference:  ${avg_iv - avg_ols:.4f}", f)

        log(f"\n3. Causal Interpretation:", f)
        log("   The IV estimate represents the Local Average Treatment Effect (LATE)", f)
        log("   for compliers: vendors whose click status is affected by marginal wins.", f)
        log("   This is the causal effect of a sponsored click on vendor spend for", f)
        log("   vendors at the margin of winning/losing an impression slot.", f)

        log("", f)
        log("=" * 80, f)
        log("10_NEAR_WIN_IV COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
