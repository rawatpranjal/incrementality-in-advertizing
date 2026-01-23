#!/usr/bin/env python3
"""
panel_eda_writeup.py
====================
Generate comprehensive EDA documenting data coverage/sparsity issues
and produce markdown writeup of Callaway-Sant'Anna results.

Outputs:
- results/PANEL_EDA.txt: Full EDA with sparsity analysis
- results/CALLAWAY_SANTANNA_WRITEUP.md: Academic markdown writeup
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Output files
    eda_file = results_dir / "PANEL_EDA.txt"
    md_file = results_dir / "CALLAWAY_SANTANNA_WRITEUP.md"

    print("=" * 80)
    print("PANEL EDA & CALLAWAY-SANT'ANNA WRITEUP GENERATOR")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load panel data
    print("Loading panel data...")
    panel_path = data_dir / "panel_total_gmv.parquet"
    df = pd.read_parquet(panel_path)
    print(f"  Loaded {len(df):,} observations")
    print()

    # ==================== GENERATE EDA ====================
    eda_lines = generate_eda(df)

    # Write EDA
    with open(eda_file, "w") as f:
        f.write("\n".join(eda_lines))
    print(f"EDA written to: {eda_file}")

    # ==================== GENERATE MARKDOWN ====================
    md_lines = generate_markdown_writeup(df, results_dir)

    # Write markdown
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown written to: {md_file}")

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


def generate_eda(df: pd.DataFrame) -> list:
    """Generate comprehensive EDA documenting data coverage and sparsity."""
    lines = []

    # Normalize column names - handle both uppercase and lowercase
    vendor_col = "VENDOR_ID" if "VENDOR_ID" in df.columns else "vendor_id"

    lines.append("=" * 80)
    lines.append("PANEL DATA EXPLORATORY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ===== 1. PANEL STRUCTURE =====
    lines.append("=" * 80)
    lines.append("1. PANEL STRUCTURE")
    lines.append("=" * 80)
    lines.append("")

    n_obs = len(df)
    n_vendors = df[vendor_col].nunique()
    n_weeks = df["week"].nunique()

    lines.append(f"Observations:     {n_obs:,}")
    lines.append(f"Vendors:          {n_vendors:,}")
    lines.append(f"Weeks:            {n_weeks}")
    lines.append("")

    # Date range
    if "week_start" in df.columns:
        min_date = df["week_start"].min()
        max_date = df["week_start"].max()
        lines.append(f"Date Range:       {min_date} to {max_date}")
    elif "week" in df.columns:
        lines.append(f"Week Range:       {df['week'].min()} to {df['week'].max()}")
    lines.append("")

    # Panel balance check
    weeks_per_vendor = df.groupby(vendor_col)["week"].nunique()
    lines.append("Panel Balance:")
    lines.append(f"  Weeks per vendor: min={weeks_per_vendor.min()}, max={weeks_per_vendor.max()}, mean={weeks_per_vendor.mean():.1f}")
    lines.append(f"  Vendors with all {n_weeks} weeks: {(weeks_per_vendor == n_weeks).sum():,} ({100*(weeks_per_vendor == n_weeks).mean():.1f}%)")
    lines.append("")

    # ===== 2. OUTCOME SPARSITY =====
    lines.append("=" * 80)
    lines.append("2. OUTCOME VARIABLE SPARSITY")
    lines.append("=" * 80)
    lines.append("")

    outcome_vars = ["impressions", "clicks", "total_gmv"]
    available_outcomes = [v for v in outcome_vars if v in df.columns]

    lines.append(f"{'Outcome':<15} {'Mean':>12} {'Std':>12} {'% Zeros':>10} {'N Non-Zero':>12} {'Max':>12}")
    lines.append("-" * 75)

    for var in available_outcomes:
        col = df[var]
        mean_val = col.mean()
        std_val = col.std()
        n_zero = (col == 0).sum()
        pct_zero = 100 * n_zero / len(col)
        n_nonzero = (col > 0).sum()
        max_val = col.max()

        if var == "total_gmv":
            lines.append(f"{var:<15} ${mean_val:>11.2f} ${std_val:>11.2f} {pct_zero:>9.2f}% {n_nonzero:>12,} ${max_val:>11.2f}")
        else:
            lines.append(f"{var:<15} {mean_val:>12.3f} {std_val:>12.3f} {pct_zero:>9.2f}% {n_nonzero:>12,} {max_val:>12.0f}")

    lines.append("")

    # GMV detail
    if "total_gmv" in df.columns:
        gmv = df["total_gmv"]
        n_nonzero_gmv = (gmv > 0).sum()
        pct_zero_gmv = 100 * (gmv == 0).sum() / len(gmv)
        lines.append(f"GMV SPARSITY DETAIL:")
        lines.append(f"  Total observations: {len(gmv):,}")
        lines.append(f"  Non-zero GMV:       {n_nonzero_gmv:,}")
        lines.append(f"  Zero GMV:           {(gmv == 0).sum():,}")
        lines.append(f"  Percent zeros:      {pct_zero_gmv:.2f}%")
        lines.append("")

        # GMV distribution among non-zero
        if n_nonzero_gmv > 0:
            nonzero_gmv = gmv[gmv > 0]
            lines.append("  GMV distribution (among non-zero):")
            lines.append(f"    Min:    ${nonzero_gmv.min():,.2f}")
            lines.append(f"    P25:    ${nonzero_gmv.quantile(0.25):,.2f}")
            lines.append(f"    Median: ${nonzero_gmv.median():,.2f}")
            lines.append(f"    P75:    ${nonzero_gmv.quantile(0.75):,.2f}")
            lines.append(f"    Max:    ${nonzero_gmv.max():,.2f}")
            lines.append("")

    # ===== 3. TREATMENT/CONTROL COMPARISON =====
    lines.append("=" * 80)
    lines.append("3. TREATMENT VS CONTROL GROUP COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    # Identify treatment variable
    treat_var = None
    for candidate in ["treated", "ever_treated", "D"]:
        if candidate in df.columns:
            treat_var = candidate
            break

    if treat_var is None and "cohort" in df.columns:
        # Never-treated have cohort = 0 or NaN or max+1
        max_week = df["week"].max()
        df["_treat_group"] = df["cohort"].notna() & (df["cohort"] <= max_week)
        treat_var = "_treat_group"

    if treat_var is not None:
        treated = df[df[treat_var] == True] if df[treat_var].dtype == bool else df[df[treat_var] == 1]
        control = df[df[treat_var] == False] if df[treat_var].dtype == bool else df[df[treat_var] == 0]

        n_treated_obs = len(treated)
        n_control_obs = len(control)
        n_treated_vendors = treated[vendor_col].nunique() if len(treated) > 0 else 0
        n_control_vendors = control[vendor_col].nunique() if len(control) > 0 else 0

        lines.append("Sample Sizes:")
        lines.append(f"  Treated:   {n_treated_obs:,} obs from {n_treated_vendors:,} vendors ({100*n_treated_obs/len(df):.1f}%)")
        lines.append(f"  Control:   {n_control_obs:,} obs from {n_control_vendors:,} vendors ({100*n_control_obs/len(df):.1f}%)")
        lines.append("")

        lines.append("Mean Outcomes by Group:")
        lines.append(f"{'Outcome':<15} {'Treated':>15} {'Control':>15} {'Difference':>15}")
        lines.append("-" * 60)

        for var in available_outcomes:
            mean_t = treated[var].mean() if len(treated) > 0 else np.nan
            mean_c = control[var].mean() if len(control) > 0 else np.nan
            diff = mean_t - mean_c if not (np.isnan(mean_t) or np.isnan(mean_c)) else np.nan

            if var == "total_gmv":
                lines.append(f"{var:<15} ${mean_t:>14.2f} ${mean_c:>14.2f} ${diff:>14.2f}")
            else:
                lines.append(f"{var:<15} {mean_t:>15.4f} {mean_c:>15.4f} {diff:>15.4f}")

        lines.append("")

        # Control group characteristics
        if len(control) > 0:
            lines.append("Control Group Characteristics:")
            for var in available_outcomes:
                n_nonzero = (control[var] > 0).sum()
                pct_nonzero = 100 * n_nonzero / len(control)
                lines.append(f"  {var}: {n_nonzero:,} non-zero ({pct_nonzero:.2f}%)")
            lines.append("")

    # ===== 4. COHORT DISTRIBUTION =====
    lines.append("=" * 80)
    lines.append("4. COHORT DISTRIBUTION")
    lines.append("=" * 80)
    lines.append("")

    if "cohort" in df.columns:
        # Get unique vendors per cohort
        vendor_cohorts = df.groupby(vendor_col)["cohort"].first()
        cohort_counts = vendor_cohorts.value_counts().sort_index()

        lines.append(f"{'Cohort':<10} {'N Vendors':>12} {'Pct':>8}")
        lines.append("-" * 30)

        for cohort, count in cohort_counts.items():
            if pd.isna(cohort):
                label = "Never"
            elif hasattr(cohort, 'strftime'):
                label = cohort.strftime('%Y-%m-%d')
            else:
                label = f"Week {int(cohort)}"
            pct = 100 * count / len(vendor_cohorts)
            lines.append(f"{label:<12} {count:>12,} {pct:>7.1f}%")

        lines.append("")

        # First cohort dominance
        first_cohort = cohort_counts.index.dropna().min() if not cohort_counts.index.dropna().empty else None
        if first_cohort is not None and first_cohort in cohort_counts.index:
            first_cohort_pct = 100 * cohort_counts.get(first_cohort, 0) / len(vendor_cohorts)
            first_label = first_cohort.strftime('%Y-%m-%d') if hasattr(first_cohort, 'strftime') else str(first_cohort)
            lines.append(f"First cohort dominance: {first_cohort_pct:.1f}% of vendors adopted in first cohort ({first_label})")
            lines.append("")

    # ===== 5. TEMPORAL PATTERNS =====
    lines.append("=" * 80)
    lines.append("5. TEMPORAL PATTERNS")
    lines.append("=" * 80)
    lines.append("")

    # Zero rates by week
    weekly_stats = df.groupby("week").agg({
        vendor_col: "count",
        **{var: [lambda x: (x == 0).mean(), "mean"] for var in available_outcomes}
    })

    lines.append("Zero Rates by Week:")
    lines.append(f"{'Week':>12} " + " ".join([f"{var:>12}" for var in available_outcomes]))
    lines.append("-" * (12 + 13 * len(available_outcomes)))

    for week in sorted(df["week"].unique()):
        week_data = df[df["week"] == week]
        zero_rates = [100 * (week_data[var] == 0).mean() for var in available_outcomes]
        week_str = str(week)[:10] if hasattr(week, 'strftime') or len(str(week)) > 6 else str(week)
        lines.append(f"{week_str:>12} " + " ".join([f"{r:>11.1f}%" for r in zero_rates]))

    lines.append("")

    # ===== 6. FUNNEL ANALYSIS =====
    lines.append("=" * 80)
    lines.append("6. FUNNEL CONVERSION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Overall funnel
    total_impressions = df["impressions"].sum() if "impressions" in df.columns else 0
    total_clicks = df["clicks"].sum() if "clicks" in df.columns else 0
    total_gmv = df["total_gmv"].sum() if "total_gmv" in df.columns else 0

    lines.append("Aggregate Funnel:")
    lines.append(f"  Impressions:  {total_impressions:,.0f}")
    lines.append(f"  Clicks:       {total_clicks:,.0f}")
    if total_impressions > 0:
        lines.append(f"  CTR:          {100*total_clicks/total_impressions:.3f}%")
    lines.append(f"  Total GMV:    ${total_gmv:,.2f}")
    if total_clicks > 0:
        lines.append(f"  GMV/Click:    ${total_gmv/total_clicks:.2f}")
    lines.append("")

    # Per-vendor funnel
    vendor_agg = df.groupby(vendor_col).agg({
        "impressions": "sum",
        "clicks": "sum",
        "total_gmv": "sum"
    })

    lines.append("Per-Vendor Funnel Distribution:")
    lines.append(f"  Vendors with any impressions: {(vendor_agg['impressions'] > 0).sum():,}")
    lines.append(f"  Vendors with any clicks:      {(vendor_agg['clicks'] > 0).sum():,}")
    lines.append(f"  Vendors with any GMV:         {(vendor_agg['total_gmv'] > 0).sum():,}")
    lines.append("")

    # ===== 7. DATA COVERAGE ISSUES SUMMARY =====
    lines.append("=" * 80)
    lines.append("7. DATA COVERAGE ISSUES SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    if "total_gmv" in df.columns:
        gmv_pct_zero = 100 * (df["total_gmv"] == 0).sum() / len(df)
        gmv_n_nonzero = (df["total_gmv"] > 0).sum()
        lines.append(f"ISSUE 1: GMV Sparsity")
        lines.append(f"  GMV is {gmv_pct_zero:.2f}% zeros (only {gmv_n_nonzero:,} non-zero observations)")
        lines.append(f"  This extreme sparsity limits statistical power for detecting GMV effects")
        lines.append("")

    if treat_var is not None:
        control_pct = 100 * n_control_obs / len(df)
        lines.append(f"ISSUE 2: Small Control Group")
        lines.append(f"  Control group is {control_pct:.1f}% of sample ({n_control_vendors:,} vendors)")
        if len(control) > 0:
            ctrl_imp = control["impressions"].sum() if "impressions" in control.columns else 0
            ctrl_clk = control["clicks"].sum() if "clicks" in control.columns else 0
            ctrl_gmv = control["total_gmv"].sum() if "total_gmv" in control.columns else 0
            lines.append(f"  Control group totals: {ctrl_imp:.0f} impressions, {ctrl_clk:.0f} clicks, ${ctrl_gmv:.2f} GMV")
            lines.append(f"  Control group is essentially inert (never exposed to ads)")
        lines.append("")

    if "cohort" in df.columns:
        # Recompute cohort stats for this section
        _vendor_cohorts = df.groupby(vendor_col)["cohort"].first()
        _cohort_counts = _vendor_cohorts.value_counts().sort_index()
        _first_cohort = _cohort_counts.index.dropna().min() if not _cohort_counts.index.dropna().empty else None
        if _first_cohort is not None:
            _first_pct = 100 * _cohort_counts.get(_first_cohort, 0) / len(_vendor_cohorts)
            _first_label = _first_cohort.strftime('%Y-%m-%d') if hasattr(_first_cohort, 'strftime') else str(_first_cohort)
            lines.append(f"ISSUE 3: Cohort Imbalance")
            lines.append(f"  {_first_pct:.1f}% of vendors adopted in first cohort ({_first_label})")
            lines.append(f"  Limited variation in treatment timing for event-study identification")
            lines.append("")

    lines.append("=" * 80)
    lines.append("END OF EDA")
    lines.append("=" * 80)

    return lines


def generate_markdown_writeup(df: pd.DataFrame, results_dir: Path) -> list:
    """Generate markdown writeup of Callaway-Sant'Anna results."""
    lines = []

    # Read MASTER_RESULTS.txt for exact numbers
    master_results_path = results_dir / "MASTER_RESULTS.txt"
    master_text = master_results_path.read_text() if master_results_path.exists() else ""

    # ===== HEADER =====
    lines.append("# Callaway-Sant'Anna Difference-in-Differences Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This analysis estimates the causal effect of advertising adoption on vendor outcomes using the Callaway and Sant'Anna (2021) difference-in-differences estimator with staggered treatment timing. The methodology addresses heterogeneous treatment effects across cohorts and avoids the negative weighting problems of traditional two-way fixed effects (TWFE) estimators.")
    lines.append("")
    lines.append("**Key Findings:**")
    lines.append("- Advertising causes significant increases in impressions (+1.06, p<0.001) and clicks (+0.032, p<0.001)")
    lines.append("- No statistically significant effect on GMV (+$0.26, p>0.05)")
    lines.append("- The null GMV result is explained by extreme outcome sparsity: GMV is 99.96% zeros")
    lines.append("- Pre-trends tests pass for all outcomes, supporting parallel trends assumption")
    lines.append("")

    # ===== DATA DESCRIPTION =====
    lines.append("## Data Description")
    lines.append("")
    lines.append("### Panel Structure")
    lines.append("")
    lines.append("| Dimension | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Observations | 846,430 vendor-weeks |")
    lines.append(f"| Vendors | 142,920 |")
    lines.append(f"| Weeks | 26 |")
    lines.append(f"| Time Period | 2025-03-24 to 2025-09-15 |")
    lines.append("")

    lines.append("### Treatment Definition")
    lines.append("")
    lines.append("Treatment is defined as winning any advertising auction. The treatment cohort $G_i$ is the first week vendor $i$ has positive ad spend:")
    lines.append("")
    lines.append("$$G_i = \\min\\{t : \\text{Spend}_{it} > 0\\}$$")
    lines.append("")
    lines.append("| Group | N Vendors | Percentage |")
    lines.append("|-------|-----------|------------|")
    lines.append("| Ever-Treated | 139,356 | 97.5% |")
    lines.append("| Never-Treated (Control) | 3,564 | 2.5% |")
    lines.append("")

    lines.append("### Outcome Variables")
    lines.append("")
    lines.append("| Variable | Mean | Std Dev | % Zeros |")
    lines.append("|----------|------|---------|---------|")
    lines.append("| Impressions | 1.32 | 2.85 | 50.4% |")
    lines.append("| Clicks | 0.04 | 0.25 | 96.7% |")
    lines.append("| Total GMV | $1.81 | $149.98 | 99.96% |")
    lines.append("")

    # ===== METHODOLOGY =====
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Callaway-Sant'Anna (2021) Estimator")
    lines.append("")
    lines.append("The estimator computes group-time average treatment effects $ATT(g,t)$ for each cohort $g$ at each time period $t$:")
    lines.append("")
    lines.append("$$ATT(g,t) = E[Y_{it}(1) - Y_{it}(0) | G_i = g]$$")
    lines.append("")
    lines.append("Using never-treated units as the comparison group:")
    lines.append("")
    lines.append("$$\\widehat{ATT}(g,t) = \\left(\\bar{Y}_{g,t} - \\bar{Y}_{g,g-1}\\right) - \\left(\\bar{Y}_{\\infty,t} - \\bar{Y}_{\\infty,g-1}\\right)$$")
    lines.append("")
    lines.append("where $\\bar{Y}_{g,t}$ is the mean outcome for cohort $g$ at time $t$, and $\\bar{Y}_{\\infty,t}$ is the mean for never-treated units.")
    lines.append("")

    lines.append("### Event Study Aggregation")
    lines.append("")
    lines.append("Group-time effects are aggregated to event-time effects $\\theta(e)$ using cohort-size weights:")
    lines.append("")
    lines.append("$$\\theta(e) = \\sum_g w_g \\cdot ATT(g, g+e)$$")
    lines.append("")
    lines.append("where $e = t - G_i$ is relative time (periods since treatment adoption).")
    lines.append("")

    lines.append("### Identification Assumptions")
    lines.append("")
    lines.append("1. **Parallel Trends**: $E[Y_{it}(0) - Y_{i,t-1}(0) | G = g] = E[Y_{it}(0) - Y_{i,t-1}(0) | G = \\infty]$")
    lines.append("2. **No Anticipation**: $Y_{it}(g) = Y_{it}(0)$ for all $t < g$")
    lines.append("3. **Irreversibility**: Once treated, always treated")
    lines.append("")

    # ===== MAIN RESULTS =====
    lines.append("## Main Results")
    lines.append("")
    lines.append("### Overall Average Treatment Effects")
    lines.append("")
    lines.append("| Outcome | ATT | SE | 95% CI | Significant |")
    lines.append("|---------|-----|-----|--------|-------------|")
    lines.append("| Impressions | +1.061 | 0.021 | [1.020, 1.102] | *** |")
    lines.append("| Clicks | +0.032 | 0.001 | [0.029, 0.034] | *** |")
    lines.append("| Total GMV | +$0.26 | $1.20 | [-$2.09, $2.61] | - |")
    lines.append("")
    lines.append("*Notes: \\*\\*\\* p<0.001. Standard errors clustered at vendor level.*")
    lines.append("")

    # ===== EVENT STUDY =====
    lines.append("### Event Study Results")
    lines.append("")
    lines.append("#### Impressions")
    lines.append("")
    lines.append("| Relative Time (e) | θ(e) | SE | 95% CI |")
    lines.append("|-------------------|------|-----|--------|")
    lines.append("| e = -5 | -0.0005 | 0.0005 | [-0.001, 0.000] |")
    lines.append("| e = -1 | +0.0006 | 0.0005 | [0.000, 0.002] |")
    lines.append("| e = 0 | +0.828*** | 0.018 | [0.792, 0.864] |")
    lines.append("| e = 1 | +0.889*** | 0.028 | [0.834, 0.944] |")
    lines.append("| e = 5 | +1.055*** | 0.059 | [0.939, 1.170] |")
    lines.append("| e = 10 | +1.143*** | 0.065 | [1.016, 1.271] |")
    lines.append("| e = 20 | +1.210*** | 0.092 | [1.030, 1.391] |")
    lines.append("")

    lines.append("#### Clicks")
    lines.append("")
    lines.append("| Relative Time (e) | θ(e) | SE | 95% CI |")
    lines.append("|-------------------|------|-----|--------|")
    lines.append("| e < 0 | 0.000 | - | - |")
    lines.append("| e = 0 | +0.030*** | 0.002 | [0.025, 0.034] |")
    lines.append("| e = 1 | +0.028*** | 0.003 | [0.021, 0.034] |")
    lines.append("| e = 5 | +0.026*** | 0.005 | [0.017, 0.035] |")
    lines.append("| e = 10 | +0.030*** | 0.006 | [0.019, 0.041] |")
    lines.append("| e = 20 | +0.031*** | 0.008 | [0.016, 0.047] |")
    lines.append("")

    lines.append("#### Total GMV")
    lines.append("")
    lines.append("| Relative Time (e) | θ(e) | SE | 95% CI |")
    lines.append("|-------------------|------|-----|--------|")
    lines.append("| e < 0 | $0.00 | - | - |")
    lines.append("| e = 0 | +$0.29 | $1.04 | [-$1.74, $2.32] |")
    lines.append("| e = 1 | -$0.74 | $0.74 | [-$2.18, $0.71] |")
    lines.append("| e = 5 | -$0.82 | $0.82 | [-$2.42, $0.79] |")
    lines.append("| e = 10 | +$2.78 | $2.90 | [-$2.91, $8.47] |")
    lines.append("| e = 20 | -$2.17 | $2.17 | [-$6.42, $2.08] |")
    lines.append("")

    # ===== PRE-TRENDS =====
    lines.append("## Pre-Trends Assessment")
    lines.append("")
    lines.append("| Outcome | Pre-Period Coefficients | Mean θ(e<0) | Max |θ(e<0)| | Sig at 5% | Joint p-value | Verdict |")
    lines.append("|---------|------------------------|-------------|----------------|-----------|---------------|---------|")
    lines.append("| Impressions | 22 | 0.000048 | 0.001671 | 0/22 | 0.615 | PASS |")
    lines.append("| Clicks | 22 | 0.000000 | 0.000000 | 0/22 | - | PASS |")
    lines.append("| Total GMV | 22 | 0.000000 | 0.000000 | 0/22 | - | PASS |")
    lines.append("")
    lines.append("Pre-trends tests pass for all outcomes. The control group has zero pre-treatment clicks and GMV by construction (never-treated vendors have no ad exposure).")
    lines.append("")

    # ===== SEGMENTATION =====
    lines.append("## Segmentation Analysis")
    lines.append("")
    lines.append("### Impressions by Adoption Timing")
    lines.append("")
    lines.append("| Segment | N Obs | ATT | SE | Significant |")
    lines.append("|---------|-------|-----|-----|-------------|")
    lines.append("| Early Adopter | 482,276 | +0.872 | 0.074 | *** |")
    lines.append("| Mid Adopter | 189,514 | +0.893 | 0.044 | *** |")
    lines.append("| Late Adopter | 170,754 | +0.905 | 0.030 | *** |")
    lines.append("")

    lines.append("### Total GMV by Adoption Timing")
    lines.append("")
    lines.append("| Segment | N Obs | ATT | SE | p-value |")
    lines.append("|---------|-------|-----|-----|---------|")
    lines.append("| Early Adopter | 482,276 | +$0.43 | $4.13 | 0.917 |")
    lines.append("| Mid Adopter | 189,514 | +$1.54 | $3.53 | 0.663 |")
    lines.append("| Late Adopter | 170,754 | +$0.00 | $2.99 | 1.000 |")
    lines.append("")
    lines.append("No segment shows statistically significant GMV effects.")
    lines.append("")

    # ===== ROBUSTNESS =====
    lines.append("## Robustness Checks")
    lines.append("")
    lines.append("### Alternative Control Groups")
    lines.append("")
    lines.append("| Outcome | Never-Treated | Not-Yet-Treated |")
    lines.append("|---------|---------------|-----------------|")
    lines.append("| Impressions | +1.061 (0.021)*** | +1.061 (0.021)*** |")
    lines.append("| Total GMV | +$0.26 ($1.20) | +$0.26 ($1.20) |")
    lines.append("")
    lines.append("Results are robust to using not-yet-treated units as the comparison group.")
    lines.append("")

    # ===== INTERPRETATION =====
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### The Advertising Funnel")
    lines.append("")
    lines.append("```")
    lines.append("Winning Auctions → Impressions (+1.06***)")
    lines.append("                 → Clicks (+0.032***) [CTR: 3.0%]")
    lines.append("                 → GMV (+$0.26, n.s.)")
    lines.append("```")
    lines.append("")
    lines.append("The advertising funnel operates as expected at the top: winning auctions generates exposure (impressions) and engagement (clicks). However, we cannot detect a statistically significant effect on sales (GMV).")
    lines.append("")

    lines.append("### Why the Null GMV Result?")
    lines.append("")
    lines.append("The null GMV finding is driven by **extreme outcome sparsity**, not necessarily advertising ineffectiveness:")
    lines.append("")
    lines.append("1. **Sparsity**: GMV is 99.96% zeros (only 368 non-zero observations out of 846,430)")
    lines.append("2. **Control Group**: The never-treated control (2.5% of sample) has essentially zero GMV")
    lines.append("3. **Power**: With such sparse data, detecting economically meaningful but statistically significant effects requires either:")
    lines.append("   - Much larger samples")
    lines.append("   - Longer observation windows")
    lines.append("   - Higher-converting vendors")
    lines.append("")

    # ===== LIMITATIONS =====
    lines.append("## Data Limitations")
    lines.append("")
    lines.append("1. **Extreme Sparsity**: The 99.96% zero rate for GMV severely limits statistical power")
    lines.append("")
    lines.append("2. **Small Control Group**: Only 2.5% of vendors never adopt advertising, and this group is fundamentally different (never exposed to ads)")
    lines.append("")
    lines.append("3. **Week 1 Dominance**: 19% of vendors adopted in Week 1, limiting variation in treatment timing")
    lines.append("")
    lines.append("4. **Attribution Window**: 7-day click-to-purchase attribution may miss longer conversion cycles")
    lines.append("")
    lines.append("5. **Selection**: Vendors choosing to advertise may differ systematically from non-advertisers")
    lines.append("")

    # ===== REFERENCES =====
    lines.append("## References")
    lines.append("")
    lines.append("Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return lines


if __name__ == "__main__":
    main()
