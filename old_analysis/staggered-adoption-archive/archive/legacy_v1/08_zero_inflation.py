#!/usr/bin/env python3
"""
EDA Q8: How severe is the Zero-Inflation?

What percentage of active VENDOR_ID × WEEK observations have zero Total GMV?
If this is extremely high (>80%), linear models (Log-GMV) will fail, and we may
need to model the "Extensive Margin" (Probability of making any sale) separately
from the "Intensive Margin" (Amount sold).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "08_zero_inflation.txt"

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
        log("EDA Q8: ZERO-INFLATION ANALYSIS", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  What % of vendor-week observations have zero outcomes?", f)
        log("  Threshold: If >80% zeros, linear models fail.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Count zero vs non-zero for each outcome variable", f)
        log("  2. Analyze by treatment status and time", f)
        log("  3. Assess need for two-part models", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel data
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_vendor_week.parquet"
        if not panel_path.exists():
            log(f"  [ERROR] File not found: {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        log(f"  Loaded {len(panel):,} vendor-week observations", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Zero-inflation by outcome
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ZERO-INFLATION BY OUTCOME VARIABLE", f)
        log("-" * 40, f)
        log("", f)

        outcome_vars = ['promoted_gmv', 'impressions', 'clicks', 'wins', 'total_spend']

        log(f"  {'Outcome':<20} {'Zeros':>12} {'Non-Zero':>12} {'Zero %':>10} {'Mean|>0':>12}", f)
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*12}", f)

        zero_stats = {}
        for var in outcome_vars:
            if var in panel.columns:
                n_zero = (panel[var] == 0).sum()
                n_nonzero = (panel[var] > 0).sum()
                zero_pct = n_zero / len(panel) * 100
                mean_nonzero = panel.loc[panel[var] > 0, var].mean()

                zero_stats[var] = {
                    'n_zero': n_zero,
                    'n_nonzero': n_nonzero,
                    'zero_pct': zero_pct,
                    'mean_nonzero': mean_nonzero
                }

                log(f"  {var:<20} {n_zero:>12,} {n_nonzero:>12,} {zero_pct:>9.1f}% {mean_nonzero:>11.2f}", f)

        log("", f)

        # -----------------------------------------------------------------
        # Zero patterns by treatment status
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ZERO-INFLATION BY TREATMENT STATUS", f)
        log("-" * 40, f)
        log("", f)

        if 'has_spend' in panel.columns:
            for status, label in [(1, 'Treated (has_spend=1)'), (0, 'Control (has_spend=0)')]:
                subset = panel[panel['has_spend'] == status]
                log(f"  {label}:", f)
                log(f"    N observations: {len(subset):,}", f)

                for var in ['promoted_gmv', 'impressions', 'clicks']:
                    if var in subset.columns:
                        zero_pct = (subset[var] == 0).sum() / len(subset) * 100
                        log(f"    {var}: {zero_pct:.1f}% zeros", f)

                log("", f)

        # -----------------------------------------------------------------
        # Joint zero patterns
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("JOINT ZERO PATTERNS", f)
        log("-" * 40, f)
        log("", f)

        # All outcomes zero
        all_zero_mask = (
            (panel['promoted_gmv'] == 0) &
            (panel['impressions'] == 0) &
            (panel['clicks'] == 0)
        )
        all_zero_pct = all_zero_mask.sum() / len(panel) * 100

        log(f"  All funnel metrics zero (GMV, impressions, clicks): {all_zero_pct:.1f}%", f)
        log("", f)

        # Has impressions but no clicks
        imp_no_click = ((panel['impressions'] > 0) & (panel['clicks'] == 0)).sum()
        imp_total = (panel['impressions'] > 0).sum()
        log(f"  Has impressions but no clicks: {imp_no_click:,} / {imp_total:,} ({imp_no_click/imp_total*100 if imp_total > 0 else 0:.1f}%)", f)

        # Has clicks but no GMV
        click_no_gmv = ((panel['clicks'] > 0) & (panel['promoted_gmv'] == 0)).sum()
        click_total = (panel['clicks'] > 0).sum()
        log(f"  Has clicks but no GMV: {click_no_gmv:,} / {click_total:,} ({click_no_gmv/click_total*100 if click_total > 0 else 0:.1f}%)", f)

        log("", f)

        # -----------------------------------------------------------------
        # Zero-inflation over time
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("ZERO-INFLATION BY WEEK", f)
        log("-" * 40, f)
        log("", f)

        weekly_zeros = panel.groupby('week').apply(
            lambda x: pd.Series({
                'n_obs': len(x),
                'gmv_zero_pct': (x['promoted_gmv'] == 0).sum() / len(x) * 100,
                'imp_zero_pct': (x['impressions'] == 0).sum() / len(x) * 100,
                'click_zero_pct': (x['clicks'] == 0).sum() / len(x) * 100
            })
        ).reset_index()

        log(f"  {'Week':<12} {'N Obs':>10} {'GMV Zero%':>12} {'Imp Zero%':>12} {'Click Zero%':>12}", f)
        log(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*12}", f)

        for _, row in weekly_zeros.iterrows():
            log(f"  {str(row['week'])[:10]:<12} {int(row['n_obs']):>10,} {row['gmv_zero_pct']:>11.1f}% {row['imp_zero_pct']:>11.1f}% {row['click_zero_pct']:>11.1f}%", f)

        log("", f)

        # -----------------------------------------------------------------
        # Vendor-level zero patterns
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR-LEVEL ZERO PATTERNS", f)
        log("-" * 40, f)
        log("", f)

        vendor_stats = panel.groupby('VENDOR_ID').agg({
            'promoted_gmv': ['sum', lambda x: (x > 0).sum()],
            'impressions': ['sum', lambda x: (x > 0).sum()],
            'week': 'count'
        }).reset_index()
        vendor_stats.columns = ['VENDOR_ID', 'total_gmv', 'weeks_with_gmv',
                                 'total_impressions', 'weeks_with_impressions', 'n_weeks']

        # Vendors who NEVER have GMV
        never_gmv = (vendor_stats['total_gmv'] == 0).sum()
        log(f"  Vendors with ZERO GMV (all weeks): {never_gmv:,} ({never_gmv/len(vendor_stats)*100:.1f}%)", f)

        # Vendors who NEVER have impressions
        never_imp = (vendor_stats['total_impressions'] == 0).sum()
        log(f"  Vendors with ZERO impressions (all weeks): {never_imp:,} ({never_imp/len(vendor_stats)*100:.1f}%)", f)

        log("", f)

        # Distribution of weeks with positive GMV
        log("  Distribution of weeks with positive GMV per vendor:", f)
        gmv_weeks_dist = vendor_stats['weeks_with_gmv'].value_counts().sort_index()
        for n_weeks, count in gmv_weeks_dist.head(10).items():
            pct = count / len(vendor_stats) * 100
            log(f"    {n_weeks} weeks: {count:,} vendors ({pct:.1f}%)", f)
        log("", f)

        # -----------------------------------------------------------------
        # GMV distribution (non-zero)
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("GMV DISTRIBUTION (NON-ZERO VALUES)", f)
        log("-" * 40, f)
        log("", f)

        nonzero_gmv = panel.loc[panel['promoted_gmv'] > 0, 'promoted_gmv']

        if len(nonzero_gmv) > 0:
            log("  Promoted GMV | GMV > 0:", f)
            log(f"    N observations: {len(nonzero_gmv):,}", f)
            log(f"    Mean: ${nonzero_gmv.mean():,.2f}", f)
            log(f"    Median: ${nonzero_gmv.median():,.2f}", f)
            log(f"    Std: ${nonzero_gmv.std():,.2f}", f)
            log("", f)

            log("  Quantiles:", f)
            for pct in [10, 25, 50, 75, 90, 95, 99]:
                val = nonzero_gmv.quantile(pct/100)
                log(f"    P{pct:02d}: ${val:,.2f}", f)
            log("", f)

            # Log-GMV distribution
            log_gmv = np.log1p(nonzero_gmv)
            log("  Log(1 + GMV) distribution:", f)
            log(f"    Mean: {log_gmv.mean():.3f}", f)
            log(f"    Std: {log_gmv.std():.3f}", f)
            log(f"    Skewness: {log_gmv.skew():.3f}", f)
            log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        gmv_zero_pct = zero_stats['promoted_gmv']['zero_pct']

        if gmv_zero_pct > 80:
            log(f"  [CRITICAL] {gmv_zero_pct:.1f}% of observations have zero GMV.", f)
            log("  Linear models on log(GMV) will fail or be heavily biased.", f)
            log("", f)
            log("  RECOMMENDATIONS:", f)
            log("    1. Use two-part model (Hurdle/Zero-Inflated):", f)
            log("       Part 1: P(GMV > 0) - Extensive margin (Probit/Logit)", f)
            log("       Part 2: E[log(GMV) | GMV > 0] - Intensive margin (OLS)", f)
            log("    2. Or use Tobit model for censored outcomes", f)
            log("    3. Or model E[GMV] directly with Poisson/Negative Binomial", f)
        elif gmv_zero_pct > 50:
            log(f"  [WARNING] {gmv_zero_pct:.1f}% of observations have zero GMV.", f)
            log("  Consider two-part model or robust transformation.", f)
            log("  Alternative: Use asinh(GMV) instead of log(GMV).", f)
        else:
            log(f"  [OK] {gmv_zero_pct:.1f}% zeros is manageable.", f)
            log("  Log(1 + GMV) transformation should work.", f)

        log("", f)

        # Funnel conversion rates
        if imp_total > 0 and click_total > 0:
            imp_to_click = click_total / imp_total * 100
            click_to_gmv = (click_total - click_no_gmv) / click_total * 100

            log("  FUNNEL CONVERSION RATES:", f)
            log(f"    Impression → Click: {imp_to_click:.2f}%", f)
            log(f"    Click → GMV: {click_to_gmv:.2f}%", f)
            log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
