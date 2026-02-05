#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Extensive vs. Intensive Margin Decomposition
Tests if ads help vendors get their first sale (extensive) vs. increase volume (intensive).
Also examines quantile effects to detect "Super Winners" in the right tail.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_PULL_DIR = BASE_DIR.parent / "data_pull" / "data"

OUTPUT_FILE = RESULTS_DIR / "17_extensive_margin.txt"

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
        log("STAGGERED ADOPTION: EXTENSIVE VS. INTENSIVE MARGIN ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Decompose treatment effects into:", f)
        log("    1. EXTENSIVE MARGIN: P(sale > 0) — did the item sell at all?", f)
        log("    2. INTENSIVE MARGIN: E[sale | sale > 0] — conditional GMV if sold", f)
        log("  Also examine quantile effects to detect 'Super Winners'.", f)
        log("", f)

        log("WHY THIS MATTERS:", f)
        log("  Average Treatment Effects (ATT) hide distributional effects.", f)
        log("  Maybe ads don't move average GMV, but they:", f)
        log("    - Create 'Super Winners' (top 5% explode)", f)
        log("    - Prevent 'Zeros' (help inventory move at all)", f)
        log("  If strategy is 'hits-based', looking at the mean is misleading.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # =================================================================
        # DATASET A: Existing 26-week Panel
        # =================================================================
        log("=" * 80, f)
        log("DATASET A: 26-WEEK PANEL (142K vendors)", f)
        log("=" * 80, f)
        log("", f)

        panel_path = DATA_DIR / "panel_with_segments.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            return

        log("Loading panel...", f)
        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])
        log(f"  Shape: {panel.shape}", f)

        # Prepare identifiers
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # -----------------------------------------------------------------
        # Create margin outcomes
        # -----------------------------------------------------------------
        log("", f)
        log("CREATING MARGIN OUTCOMES:", f)
        log("-" * 40, f)

        # Binary outcome for extensive margin
        panel['any_sale'] = (panel['promoted_gmv'] > 0).astype(int)

        # Log outcome for intensive margin (already exists)
        # panel['log_promoted_gmv'] exists

        # Summary stats
        n_total = len(panel)
        n_any_sale = panel['any_sale'].sum()
        pct_any_sale = n_any_sale / n_total * 100

        log(f"  Total vendor-week observations: {n_total:,}", f)
        log(f"  Observations with any sale:     {n_any_sale:,} ({pct_any_sale:.2f}%)", f)
        log("", f)

        # By treatment status
        log("  BY TREATMENT STATUS:", f)
        for treated in [0, 1]:
            subset = panel[panel['treated'] == treated]
            n_obs = len(subset)
            n_sale = subset['any_sale'].sum()
            pct = n_sale / n_obs * 100 if n_obs > 0 else 0
            status = "Treated" if treated == 1 else "Control"
            log(f"    {status}: {n_sale:,} / {n_obs:,} ({pct:.2f}%)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 1: Extensive Margin DiD
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 1: EXTENSIVE MARGIN (Linear Probability Model)", f)
        log("-" * 40, f)
        log("", f)

        log("MODEL: P(any_sale = 1) = β * treated + α_vendor + γ_week + ε", f)
        log("  β = effect of treatment on PROBABILITY of having any sale", f)
        log("", f)

        try:
            model_extensive = pf.feols(
                "any_sale ~ treated | vendor_str + week_str",
                data=panel,
                vcov={'CRV1': 'vendor_str'}
            )

            att_ext = model_extensive.coef()['treated']
            se_ext = model_extensive.se()['treated']
            pval_ext = model_extensive.pvalue()['treated']
            tstat_ext = att_ext / se_ext

            log(f"  ATT (extensive):  {att_ext:.6f}", f)
            log(f"  SE:               {se_ext:.6f}", f)
            log(f"  t-stat:           {tstat_ext:.4f}", f)
            log(f"  p-value:          {pval_ext:.4f}", f)
            log("", f)

            # Interpretation
            pp_change = att_ext * 100
            log(f"  INTERPRETATION:", f)
            log(f"    Treatment increases probability of sale by {pp_change:.4f} percentage points", f)

            # Relative to baseline
            baseline_prob = panel[panel['treated'] == 0]['any_sale'].mean() * 100
            log(f"    Baseline probability (control): {baseline_prob:.2f}%", f)

            if baseline_prob > 0:
                pct_change = (att_ext * 100 / baseline_prob) * 100
                log(f"    Relative increase: {pct_change:.2f}%", f)

        except Exception as e:
            log(f"  ERROR: {str(e)}", f)
            att_ext = np.nan

        log("", f)

        # -----------------------------------------------------------------
        # Section 2: Intensive Margin DiD
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 2: INTENSIVE MARGIN (Conditional on Sale > 0)", f)
        log("-" * 40, f)
        log("", f)

        log("MODEL: log(GMV | GMV > 0) = β * treated + α_vendor + γ_week + ε", f)
        log("  β = effect of treatment on LOG GMV conditional on selling", f)
        log("", f)

        # Filter to positive sales only
        panel_positive = panel[panel['promoted_gmv'] > 0].copy()
        log(f"  Observations with positive sales: {len(panel_positive):,}", f)
        log(f"  Treated: {(panel_positive['treated']==1).sum():,}", f)
        log(f"  Control: {(panel_positive['treated']==0).sum():,}", f)
        log("", f)

        try:
            model_intensive = pf.feols(
                "log_promoted_gmv ~ treated | vendor_str + week_str",
                data=panel_positive,
                vcov={'CRV1': 'vendor_str'}
            )

            att_int = model_intensive.coef()['treated']
            se_int = model_intensive.se()['treated']
            pval_int = model_intensive.pvalue()['treated']
            tstat_int = att_int / se_int

            log(f"  ATT (intensive):  {att_int:.6f}", f)
            log(f"  SE:               {se_int:.6f}", f)
            log(f"  t-stat:           {tstat_int:.4f}", f)
            log(f"  p-value:          {pval_int:.4f}", f)
            log("", f)

            # Interpretation
            pct_change = (np.exp(att_int) - 1) * 100
            log(f"  INTERPRETATION:", f)
            log(f"    Treatment increases GMV by {pct_change:.2f}% (conditional on sale)", f)

        except Exception as e:
            log(f"  ERROR: {str(e)}", f)
            att_int = np.nan

        log("", f)

        # -----------------------------------------------------------------
        # Section 3: Quantile Effects
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 3: QUANTILE EFFECTS (Distribution Analysis)", f)
        log("-" * 40, f)
        log("", f)

        log("QUESTION: Are there 'Super Winners' in the right tail hidden by the mean?", f)
        log("", f)

        # Compare distributions
        treated_gmv = panel[panel['treated'] == 1]['promoted_gmv']
        control_gmv = panel[panel['treated'] == 0]['promoted_gmv']

        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        log(f"  {'Quantile':<12} {'Treated':<15} {'Control':<15} {'Difference':<15} {'% Diff':<12}", f)
        log(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12}", f)

        for q in quantiles:
            t_val = treated_gmv.quantile(q)
            c_val = control_gmv.quantile(q)
            diff = t_val - c_val
            pct_diff = (diff / c_val * 100) if c_val > 0 else np.nan

            marker = " <- Super Winners?" if q >= 0.90 else ""
            log(f"  Q{int(q*100):<10} {t_val:<15.2f} {c_val:<15.2f} {diff:<15.2f} {pct_diff:<12.1f}%{marker}", f)

        log("", f)

        # Positive GMV quantiles
        log("QUANTILES (Conditional on GMV > 0):", f)
        treated_pos = panel[(panel['treated'] == 1) & (panel['promoted_gmv'] > 0)]['promoted_gmv']
        control_pos = panel[(panel['treated'] == 0) & (panel['promoted_gmv'] > 0)]['promoted_gmv']

        log(f"  {'Quantile':<12} {'Treated':<15} {'Control':<15} {'Difference':<15} {'% Diff':<12}", f)
        log(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*12}", f)

        for q in quantiles:
            t_val = treated_pos.quantile(q) if len(treated_pos) > 0 else np.nan
            c_val = control_pos.quantile(q) if len(control_pos) > 0 else np.nan
            diff = t_val - c_val if not (np.isnan(t_val) or np.isnan(c_val)) else np.nan
            pct_diff = (diff / c_val * 100) if c_val > 0 else np.nan

            t_str = f"{t_val:.2f}" if not np.isnan(t_val) else "N/A"
            c_str = f"{c_val:.2f}" if not np.isnan(c_val) else "N/A"
            d_str = f"{diff:.2f}" if not np.isnan(diff) else "N/A"
            p_str = f"{pct_diff:.1f}%" if not np.isnan(pct_diff) else "N/A"

            log(f"  Q{int(q*100):<10} {t_str:<15} {c_str:<15} {d_str:<15} {p_str:<12}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 4: Click Analysis (Alternative Outcome)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 4: CLICK MARGIN ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("Alternative outcome: Clicks (higher power than GMV)", f)
        log("", f)

        if 'clicks' in panel.columns:
            panel['any_click'] = (panel['clicks'] > 0).astype(int)
            panel['log_clicks'] = np.log1p(panel['clicks'])

            # Click extensive margin
            try:
                model_click_ext = pf.feols(
                    "any_click ~ treated | vendor_str + week_str",
                    data=panel,
                    vcov={'CRV1': 'vendor_str'}
                )

                att_click_ext = model_click_ext.coef()['treated']
                se_click_ext = model_click_ext.se()['treated']
                pval_click_ext = model_click_ext.pvalue()['treated']

                log(f"  EXTENSIVE MARGIN (any click):", f)
                log(f"    ATT: {att_click_ext:.6f} (SE: {se_click_ext:.6f}, p={pval_click_ext:.4f})", f)

            except Exception as e:
                log(f"  ERROR: {str(e)}", f)

            # Click intensive margin
            panel_clicks_pos = panel[panel['clicks'] > 0].copy()
            if len(panel_clicks_pos) > 100:
                try:
                    model_click_int = pf.feols(
                        "log_clicks ~ treated | vendor_str + week_str",
                        data=panel_clicks_pos,
                        vcov={'CRV1': 'vendor_str'}
                    )

                    att_click_int = model_click_int.coef()['treated']
                    se_click_int = model_click_int.se()['treated']

                    log(f"  INTENSIVE MARGIN (clicks | clicks > 0):", f)
                    log(f"    ATT: {att_click_int:.6f} (SE: {se_click_int:.6f})", f)

                except Exception as e:
                    log(f"  ERROR: {str(e)}", f)
        else:
            log("  Clicks column not available", f)

        log("", f)

        # -----------------------------------------------------------------
        # Section 5: By Segment Analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SECTION 5: MARGIN EFFECTS BY SEGMENT", f)
        log("-" * 40, f)
        log("", f)

        log("Do different vendor segments show different margin effects?", f)
        log("", f)

        if 'persona' in panel.columns:
            log("EXTENSIVE MARGIN BY PERSONA:", f)
            log(f"  {'Persona':<25} {'ATT':<12} {'SE':<12} {'p-value':<10} {'Significant?':<12}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*12}", f)

            for persona in tqdm(panel['persona'].unique(), desc="Analyzing personas"):
                seg_panel = panel[panel['persona'] == persona]

                if seg_panel['treated'].nunique() < 2 or len(seg_panel) < 1000:
                    continue

                try:
                    model_seg = pf.feols(
                        "any_sale ~ treated | vendor_str + week_str",
                        data=seg_panel,
                        vcov={'CRV1': 'vendor_str'}
                    )

                    att = model_seg.coef()['treated']
                    se = model_seg.se()['treated']
                    pval = model_seg.pvalue()['treated']
                    sig = "*" if pval < 0.05 else ""

                    log(f"  {persona:<25} {att:<12.6f} {se:<12.6f} {pval:<10.4f} {sig:<12}", f)

                except Exception as e:
                    log(f"  {persona:<25} ERROR: {str(e)[:30]}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("MARGIN DECOMPOSITION SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        log("DATASET A (26-week panel):", f)
        log(f"  {'Margin':<25} {'ATT':<12} {'SE':<12} {'Interpretation':<30}", f)
        log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*30}", f)

        if not np.isnan(att_ext):
            interp = f"{att_ext*100:.4f} pp more likely to sell"
            log(f"  {'Extensive (P>0)':<25} {att_ext:<12.6f} {se_ext:<12.6f} {interp:<30}", f)

        if not np.isnan(att_int):
            pct = (np.exp(att_int) - 1) * 100
            interp = f"{pct:.2f}% more GMV if sold"
            log(f"  {'Intensive (E|>0)':<25} {att_int:<12.6f} {se_int:<12.6f} {interp:<30}", f)

        log("", f)

        log("KEY FINDINGS:", f)

        if not np.isnan(att_ext):
            if att_ext > 0 and pval_ext < 0.05:
                log("  1. EXTENSIVE MARGIN: Advertising INCREASES probability of sale (significant)", f)
            elif att_ext > 0:
                log("  1. EXTENSIVE MARGIN: Positive but NOT statistically significant", f)
            else:
                log("  1. EXTENSIVE MARGIN: No effect on probability of sale", f)

        if not np.isnan(att_int):
            if att_int > 0 and pval_int < 0.05:
                log("  2. INTENSIVE MARGIN: Advertising INCREASES GMV conditional on sale (significant)", f)
            elif att_int > 0:
                log("  2. INTENSIVE MARGIN: Positive but NOT statistically significant", f)
            else:
                log("  2. INTENSIVE MARGIN: No effect on conditional GMV", f)

        log("", f)
        log("  3. QUANTILE EFFECTS:", f)

        # Check right tail
        if len(treated_pos) > 0 and len(control_pos) > 0:
            q90_diff = treated_pos.quantile(0.90) - control_pos.quantile(0.90)
            q99_diff = treated_pos.quantile(0.99) - control_pos.quantile(0.99)

            if q90_diff > 0 and q99_diff > 0:
                log("     Right tail (Q90+) is HIGHER for treated → possible 'Super Winners'", f)
            elif q90_diff < 0 or q99_diff < 0:
                log("     Right tail (Q90+) is LOWER for treated → no 'Super Winners' effect", f)
            else:
                log("     Right tail effects unclear", f)

        log("", f)
        log("=" * 80, f)
        log("EXTENSIVE/INTENSIVE MARGIN ANALYSIS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
