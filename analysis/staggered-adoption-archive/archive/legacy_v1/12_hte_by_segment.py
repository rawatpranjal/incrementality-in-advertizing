#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Heterogeneous Treatment Effects by Segment
Runs TWFE DiD separately for each vendor segment using pyfixest.
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

OUTPUT_FILE = RESULTS_DIR / "12_hte_by_segment.txt"

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
        log("STAGGERED ADOPTION: HETEROGENEOUS TREATMENT EFFECTS BY SEGMENT", f)
        log("=" * 80, f)
        log("", f)

        log("PURPOSE:", f)
        log("  Estimate treatment effects separately for each vendor segment.", f)
        log("  This allows us to answer:", f)
        log("    - For whom does advertising work best?", f)
        log("    - Are treatment effects larger for Power Sellers vs Casual Sellers?", f)
        log("    - Do pre-trends hold within segments?", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  Two-Way Fixed Effects (TWFE) DiD", f)
        log("  Model: Y_it = beta * treated_it + vendor_i + week_t + epsilon_it", f)
        log("  Run separately for each persona segment", f)
        log("  Compare ATT across segments", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel with segments
        # -----------------------------------------------------------------
        log("LOADING PANEL WITH SEGMENTS", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_segments.parquet"

        if not panel_path.exists():
            log(f"  ERROR: Panel not found at {panel_path}", f)
            log("  Run 11_vendor_segmentation.py first", f)
            return

        panel = pd.read_parquet(panel_path)
        panel['week'] = pd.to_datetime(panel['week'])

        log(f"  Panel shape: {panel.shape}", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # Check segment columns
        segment_cols = ['persona', 'activity_quartile', 'is_active']
        for col in segment_cols:
            if col in panel.columns:
                log(f"  {col} segments: {panel[col].nunique()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Prepare for estimation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("PREPARING DATA FOR ESTIMATION", f)
        log("-" * 40, f)

        # Create string IDs for pyfixest
        panel['vendor_str'] = panel['VENDOR_ID'].astype(str)
        panel['week_str'] = panel['week'].astype(str)

        # Outcomes to analyze
        outcomes = ['log_promoted_gmv', 'clicks', 'impressions']

        log(f"  Outcomes: {outcomes}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Overall TWFE (for reference)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("OVERALL TWFE ESTIMATION (REFERENCE)", f)
        log("-" * 40, f)
        log("", f)

        try:
            overall_model = pf.feols(
                "log_promoted_gmv ~ treated | vendor_str + week_str",
                data=panel,
                vcov={'CRV1': 'vendor_str'}
            )

            log(f"  Overall ATT (log_promoted_gmv):", f)
            log(f"    Coefficient: {overall_model.coef()['treated']:.6f}", f)
            log(f"    Std Error: {overall_model.se()['treated']:.6f}", f)
            log(f"    t-stat: {overall_model.tstat()['treated']:.4f}", f)
            log(f"    p-value: {overall_model.pvalue()['treated']:.6f}", f)
            log(f"    N observations: {overall_model._N:,}", f)
            log("", f)

            overall_att = overall_model.coef()['treated']
        except Exception as e:
            log(f"  ERROR: {str(e)}", f)
            overall_att = np.nan

        # -----------------------------------------------------------------
        # HTE Analysis by Persona
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("HTE ANALYSIS BY PERSONA", f)
        log("-" * 40, f)

        if 'persona' not in panel.columns:
            log("  ERROR: persona column not found", f)
            return

        personas = panel['persona'].unique()
        log(f"  Personas to analyze: {list(personas)}", f)
        log("", f)

        # Store results
        hte_results = []

        for persona in tqdm(personas, desc="Processing personas"):
            log(f"\n  {'='*60}", f)
            log(f"  PERSONA: {persona}", f)
            log(f"  {'='*60}", f)

            # Subset to segment
            segment_panel = panel[panel['persona'] == persona].copy()

            n_vendors = segment_panel['VENDOR_ID'].nunique()
            n_obs = len(segment_panel)
            n_treated = segment_panel[segment_panel['treated'] == 1]['VENDOR_ID'].nunique()
            n_never_treated = segment_panel[segment_panel['cohort_week'].isna()]['VENDOR_ID'].nunique()

            log(f"    Observations: {n_obs:,}", f)
            log(f"    Unique vendors: {n_vendors:,}", f)
            log(f"    Treated vendors: {n_treated:,}", f)
            log(f"    Never-treated vendors: {n_never_treated:,}", f)
            log("", f)

            # Skip if insufficient variation
            if n_vendors < 50:
                log(f"    SKIPPING: Too few vendors ({n_vendors})", f)
                hte_results.append({
                    'segment': persona,
                    'segment_type': 'persona',
                    'n_vendors': n_vendors,
                    'n_obs': n_obs,
                    'n_treated': n_treated,
                    'n_never_treated': n_never_treated,
                    'att': np.nan,
                    'se': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan,
                })
                continue

            # Check variance in treatment
            if segment_panel['treated'].nunique() < 2:
                log(f"    SKIPPING: No variation in treatment", f)
                hte_results.append({
                    'segment': persona,
                    'segment_type': 'persona',
                    'n_vendors': n_vendors,
                    'n_obs': n_obs,
                    'n_treated': n_treated,
                    'n_never_treated': n_never_treated,
                    'att': np.nan,
                    'se': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan,
                })
                continue

            # Run TWFE for primary outcome
            try:
                log(f"    Estimating TWFE for log_promoted_gmv...", f)

                model = pf.feols(
                    "log_promoted_gmv ~ treated | vendor_str + week_str",
                    data=segment_panel,
                    vcov={'CRV1': 'vendor_str'}
                )

                att = model.coef()['treated']
                se = model.se()['treated']
                t_stat = model.tstat()['treated']
                p_val = model.pvalue()['treated']

                log(f"    Results:", f)
                log(f"      ATT: {att:.6f}", f)
                log(f"      SE: {se:.6f}", f)
                log(f"      t-stat: {t_stat:.4f}", f)
                log(f"      p-value: {p_val:.6f}", f)
                log("", f)

                # Interpret
                pct_effect = (np.exp(att) - 1) * 100
                log(f"    Interpretation:", f)
                log(f"      Advertising increases promoted GMV by approximately {pct_effect:.2f}%", f)

                hte_results.append({
                    'segment': persona,
                    'segment_type': 'persona',
                    'n_vendors': n_vendors,
                    'n_obs': n_obs,
                    'n_treated': n_treated,
                    'n_never_treated': n_never_treated,
                    'att': att,
                    'se': se,
                    't_stat': t_stat,
                    'p_value': p_val,
                })

            except Exception as e:
                log(f"    ERROR: {str(e)}", f)
                hte_results.append({
                    'segment': persona,
                    'segment_type': 'persona',
                    'n_vendors': n_vendors,
                    'n_obs': n_obs,
                    'n_treated': n_treated,
                    'n_never_treated': n_never_treated,
                    'att': np.nan,
                    'se': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan,
                })

        log("", f)

        # -----------------------------------------------------------------
        # HTE Analysis by Activity Quartile
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("HTE ANALYSIS BY ACTIVITY QUARTILE", f)
        log("-" * 40, f)

        if 'activity_quartile' in panel.columns:
            activity_quartiles = panel['activity_quartile'].unique()

            for quartile in tqdm(activity_quartiles, desc="Processing activity quartiles"):
                log(f"\n  Activity Quartile: {quartile}", f)

                segment_panel = panel[panel['activity_quartile'] == quartile].copy()

                n_vendors = segment_panel['VENDOR_ID'].nunique()
                n_obs = len(segment_panel)

                log(f"    Observations: {n_obs:,}, Vendors: {n_vendors:,}", f)

                if n_vendors < 50 or segment_panel['treated'].nunique() < 2:
                    log(f"    SKIPPING: Insufficient variation", f)
                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'activity_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': np.nan,
                        'n_never_treated': np.nan,
                        'att': np.nan,
                        'se': np.nan,
                        't_stat': np.nan,
                        'p_value': np.nan,
                    })
                    continue

                try:
                    model = pf.feols(
                        "log_promoted_gmv ~ treated | vendor_str + week_str",
                        data=segment_panel,
                        vcov={'CRV1': 'vendor_str'}
                    )

                    att = model.coef()['treated']
                    se = model.se()['treated']

                    log(f"    ATT: {att:.6f} (SE: {se:.6f})", f)

                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'activity_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': segment_panel[segment_panel['treated'] == 1]['VENDOR_ID'].nunique(),
                        'n_never_treated': segment_panel[segment_panel['cohort_week'].isna()]['VENDOR_ID'].nunique(),
                        'att': att,
                        'se': se,
                        't_stat': model.tstat()['treated'],
                        'p_value': model.pvalue()['treated'],
                    })

                except Exception as e:
                    log(f"    ERROR: {str(e)}", f)
                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'activity_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': np.nan,
                        'n_never_treated': np.nan,
                        'att': np.nan,
                        'se': np.nan,
                        't_stat': np.nan,
                        'p_value': np.nan,
                    })

        log("", f)

        # -----------------------------------------------------------------
        # HTE Analysis by Price Quartile
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("HTE ANALYSIS BY PRICE QUARTILE", f)
        log("-" * 40, f)

        if 'price_quartile' in panel.columns:
            price_quartiles = panel['price_quartile'].unique()

            for quartile in tqdm(price_quartiles, desc="Processing price quartiles"):
                log(f"\n  Price Quartile: {quartile}", f)

                segment_panel = panel[panel['price_quartile'] == quartile].copy()

                n_vendors = segment_panel['VENDOR_ID'].nunique()
                n_obs = len(segment_panel)

                log(f"    Observations: {n_obs:,}, Vendors: {n_vendors:,}", f)

                if n_vendors < 50 or segment_panel['treated'].nunique() < 2:
                    log(f"    SKIPPING: Insufficient variation", f)
                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'price_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': np.nan,
                        'n_never_treated': np.nan,
                        'att': np.nan,
                        'se': np.nan,
                        't_stat': np.nan,
                        'p_value': np.nan,
                    })
                    continue

                try:
                    model = pf.feols(
                        "log_promoted_gmv ~ treated | vendor_str + week_str",
                        data=segment_panel,
                        vcov={'CRV1': 'vendor_str'}
                    )

                    att = model.coef()['treated']
                    se = model.se()['treated']

                    log(f"    ATT: {att:.6f} (SE: {se:.6f})", f)

                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'price_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': segment_panel[segment_panel['treated'] == 1]['VENDOR_ID'].nunique(),
                        'n_never_treated': segment_panel[segment_panel['cohort_week'].isna()]['VENDOR_ID'].nunique(),
                        'att': att,
                        'se': se,
                        't_stat': model.tstat()['treated'],
                        'p_value': model.pvalue()['treated'],
                    })

                except Exception as e:
                    log(f"    ERROR: {str(e)}", f)
                    hte_results.append({
                        'segment': quartile,
                        'segment_type': 'price_quartile',
                        'n_vendors': n_vendors,
                        'n_obs': n_obs,
                        'n_treated': np.nan,
                        'n_never_treated': np.nan,
                        'att': np.nan,
                        'se': np.nan,
                        't_stat': np.nan,
                        'p_value': np.nan,
                    })

        log("", f)

        # -----------------------------------------------------------------
        # Summary comparison across segments
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("HTE SUMMARY: COMPARISON ACROSS SEGMENTS", f)
        log("-" * 40, f)
        log("", f)

        results_df = pd.DataFrame(hte_results)

        # Print by segment type
        for seg_type in results_df['segment_type'].unique():
            log(f"  {seg_type.upper()}:", f)
            log(f"  {'Segment':<25} {'N Vendors':<12} {'ATT':<12} {'SE':<12} {'p-value':<12} {'% Effect':<12}", f)
            log(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)

            seg_results = results_df[results_df['segment_type'] == seg_type]
            for _, row in seg_results.iterrows():
                att_str = f"{row['att']:.6f}" if not np.isnan(row['att']) else "N/A"
                se_str = f"{row['se']:.6f}" if not np.isnan(row['se']) else "N/A"
                pval_str = f"{row['p_value']:.6f}" if not np.isnan(row['p_value']) else "N/A"

                if not np.isnan(row['att']):
                    pct_effect = (np.exp(row['att']) - 1) * 100
                    pct_str = f"{pct_effect:.2f}%"
                else:
                    pct_str = "N/A"

                log(f"  {row['segment']:<25} {row['n_vendors']:<12,} {att_str:<12} {se_str:<12} {pval_str:<12} {pct_str:<12}", f)

            log("", f)

        # Rank segments by ATT
        valid_results = results_df[results_df['att'].notna()].copy()
        if len(valid_results) > 0:
            valid_results = valid_results.sort_values('att', ascending=False)

            log("  ALL SEGMENTS RANKED BY TREATMENT EFFECT:", f)
            log(f"  {'Rank':<6} {'Segment':<25} {'Type':<20} {'ATT':<12} {'% Effect':<12}", f)
            log(f"  {'-'*6} {'-'*25} {'-'*20} {'-'*12} {'-'*12}", f)

            for rank, (_, row) in enumerate(valid_results.iterrows(), 1):
                pct_effect = (np.exp(row['att']) - 1) * 100
                log(f"  {rank:<6} {row['segment']:<25} {row['segment_type']:<20} {row['att']:<12.6f} {pct_effect:<12.2f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Statistical comparison
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("STATISTICAL COMPARISON ACROSS PERSONAS", f)
        log("-" * 40, f)
        log("", f)

        persona_results = results_df[(results_df['segment_type'] == 'persona') & (results_df['att'].notna())]

        if len(persona_results) >= 2:
            max_row = persona_results.loc[persona_results['att'].idxmax()]
            min_row = persona_results.loc[persona_results['att'].idxmin()]

            log(f"  Highest ATT: {max_row['segment']}", f)
            log(f"    ATT = {max_row['att']:.6f} (SE = {max_row['se']:.6f})", f)
            log(f"    Effect: {(np.exp(max_row['att']) - 1) * 100:.2f}% increase in promoted GMV", f)
            log("", f)

            log(f"  Lowest ATT: {min_row['segment']}", f)
            log(f"    ATT = {min_row['att']:.6f} (SE = {min_row['se']:.6f})", f)
            log(f"    Effect: {(np.exp(min_row['att']) - 1) * 100:.2f}% increase in promoted GMV", f)
            log("", f)

            # Test for difference
            att_diff = max_row['att'] - min_row['att']
            se_diff = np.sqrt(max_row['se']**2 + min_row['se']**2)
            t_diff = att_diff / se_diff
            p_diff = 2 * (1 - min(abs(t_diff) / 4, 1))  # Approximate

            log(f"  Difference in ATT:", f)
            log(f"    {max_row['segment']} - {min_row['segment']} = {att_diff:.6f}", f)
            log(f"    SE of difference: {se_diff:.6f}", f)
            log(f"    t-statistic: {t_diff:.4f}", f)
            log(f"    Approximate p-value: {p_diff:.4f}", f)

            if p_diff < 0.05:
                log(f"    CONCLUSION: Statistically significant difference at 5% level", f)
            else:
                log(f"    CONCLUSION: Difference not statistically significant at 5% level", f)

        log("", f)

        # -----------------------------------------------------------------
        # Save results
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("SAVING RESULTS", f)
        log("-" * 40, f)

        output_path = DATA_DIR / "hte_by_segment.parquet"
        results_df.to_parquet(output_path, index=False)
        log(f"  Saved to: {output_path}", f)
        log("", f)

        log("=" * 80, f)
        log("HTE BY SEGMENT ANALYSIS COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
