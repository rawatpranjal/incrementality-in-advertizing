#!/usr/bin/env python3
"""
EDA Q10: What is the baseline "Organic Cannibalization" risk?

For vendors who adopt ads, what is the correlation between Promoted_Impressions
and Organic_GMV in the post-treatment period? A strong negative correlation
suggests we are just paying to convert users who would have bought anyway.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "10_organic_cannibalization.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# SNOWFLAKE CONNECTION
# =============================================================================
def get_snowflake_connection():
    """Establish Snowflake connection using environment variables."""
    try:
        import snowflake.connector
        load_dotenv()

        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            database='INCREMENTALITY',
            schema='INCREMENTALITY_RESEARCH'
        )
        return conn
    except Exception as e:
        return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q10: ORGANIC CANNIBALIZATION RISK", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  For treated vendors, is there a negative correlation between", f)
        log("  promoted activity and organic sales?", f)
        log("  Negative correlation = Cannibalization (paying for organic buyers).", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Compute organic_gmv = total_gmv - promoted_gmv", f)
        log("  2. For treated vendors post-treatment:", f)
        log("     Correlate impressions/clicks with organic_gmv", f)
        log("  3. Estimate cannibalization coefficient", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Load panel data
        # -----------------------------------------------------------------
        log("LOADING PANEL DATA", f)
        log("-" * 40, f)

        panel_path = DATA_DIR / "panel_with_cohorts.parquet"
        if not panel_path.exists():
            log(f"  [ERROR] File not found: {panel_path}", f)
            return

        panel = pd.read_parquet(panel_path)
        log(f"  Loaded {len(panel):,} vendor-week observations", f)
        log(f"  Unique vendors: {panel['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # Check for organic_gmv
        if 'organic_gmv' not in panel.columns:
            log("  [NOTE] organic_gmv not in panel - will need to compute from Snowflake.", f)
            log("  For now, using promoted_gmv analysis as proxy.", f)
            log("", f)
            has_organic = False
        else:
            has_organic = True

        # -----------------------------------------------------------------
        # Filter to treated vendors in post-treatment period
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("POST-TREATMENT SAMPLE", f)
        log("-" * 40, f)
        log("", f)

        # Filter to treated vendors in post-treatment
        if 'relative_week' in panel.columns:
            post_treatment = panel[
                (panel['treated'] == 1) &
                (panel['relative_week'] >= 0)
            ].copy()
        else:
            post_treatment = panel[
                (panel['has_spend'] == 1)
            ].copy()

        log(f"  Post-treatment observations: {len(post_treatment):,}", f)
        log(f"  Unique vendors: {post_treatment['VENDOR_ID'].nunique():,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Correlation analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("CORRELATION ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Compute correlations between promoted activity and GMV
        corr_vars = ['impressions', 'clicks', 'wins', 'promoted_gmv']

        log("  CORRELATION MATRIX (post-treatment observations):", f)
        log("", f)

        corr_matrix = post_treatment[corr_vars].corr()

        # Build header row
        header = f"  {'':<18}"
        for col in corr_vars:
            header += f"{col[:12]:>14}"
        log(header, f)

        # Build data rows
        for row in corr_vars:
            row_str = f"  {row:<18}"
            for col in corr_vars:
                row_str += f"{corr_matrix.loc[row, col]:>14.3f}"
            log(row_str, f)

        log("", f)

        # -----------------------------------------------------------------
        # If organic_gmv available, analyze cannibalization directly
        # -----------------------------------------------------------------
        if has_organic:
            log("=" * 80, f)
            log("ORGANIC VS PROMOTED CORRELATION", f)
            log("-" * 40, f)
            log("", f)

            corr_organic_imp = post_treatment['impressions'].corr(post_treatment['organic_gmv'])
            corr_organic_clicks = post_treatment['clicks'].corr(post_treatment['organic_gmv'])
            corr_organic_promoted = post_treatment['promoted_gmv'].corr(post_treatment['organic_gmv'])

            log(f"  Correlation with Organic GMV:", f)
            log(f"    Impressions: {corr_organic_imp:.4f}", f)
            log(f"    Clicks: {corr_organic_clicks:.4f}", f)
            log(f"    Promoted GMV: {corr_organic_promoted:.4f}", f)
            log("", f)

        # -----------------------------------------------------------------
        # Alternative: Estimate cannibalization from panel regression
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("CANNIBALIZATION PROXY ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        log("  Since organic_gmv requires fresh data pull,", f)
        log("  we analyze patterns that suggest cannibalization:", f)
        log("", f)

        # Check if high impression periods have lower conversion rates
        post_treatment['click_rate'] = post_treatment['clicks'] / post_treatment['impressions'].replace(0, np.nan)
        post_treatment['conversion_rate'] = post_treatment['promoted_gmv'] / post_treatment['clicks'].replace(0, np.nan)

        # Correlation between activity level and efficiency
        valid_cr = post_treatment.dropna(subset=['click_rate', 'conversion_rate'])

        if len(valid_cr) > 0:
            log("  EFFICIENCY METRICS:", f)
            log(f"    N observations with valid rates: {len(valid_cr):,}", f)
            log("", f)

            log("  Click Rate (clicks/impressions):", f)
            log(f"    Mean: {valid_cr['click_rate'].mean()*100:.3f}%", f)
            log(f"    Median: {valid_cr['click_rate'].median()*100:.3f}%", f)
            log("", f)

            log("  Conversion Rate (GMV/clicks):", f)
            log(f"    Mean: ${valid_cr['conversion_rate'].mean():,.2f}", f)
            log(f"    Median: ${valid_cr['conversion_rate'].median():,.2f}", f)
            log("", f)

            # Check if more impressions = lower conversion (diminishing returns)
            # This could indicate we're reaching less-qualified users
            corr_imp_conv = valid_cr['impressions'].corr(valid_cr['conversion_rate'])
            log(f"  Correlation(impressions, conversion_rate): {corr_imp_conv:.4f}", f)

            if corr_imp_conv < -0.1:
                log("    --> Negative correlation suggests diminishing returns", f)
                log("    --> As impressions increase, conversion efficiency decreases", f)
            elif corr_imp_conv > 0.1:
                log("    --> Positive correlation suggests economies of scale", f)
            else:
                log("    --> Weak correlation, no clear pattern", f)

            log("", f)

        # -----------------------------------------------------------------
        # Vendor-level cannibalization analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR-LEVEL EFFICIENCY ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Aggregate to vendor level
        vendor_post = post_treatment.groupby('VENDOR_ID').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'promoted_gmv': 'sum',
            'relative_week': 'count'
        }).reset_index()
        vendor_post.columns = ['VENDOR_ID', 'total_impressions', 'total_clicks',
                                'total_gmv', 'n_weeks']

        # Compute efficiency metrics
        vendor_post['click_rate'] = vendor_post['total_clicks'] / vendor_post['total_impressions'].replace(0, np.nan)
        vendor_post['gmv_per_click'] = vendor_post['total_gmv'] / vendor_post['total_clicks'].replace(0, np.nan)
        vendor_post['gmv_per_imp'] = vendor_post['total_gmv'] / vendor_post['total_impressions'].replace(0, np.nan)

        valid_vendors = vendor_post.dropna(subset=['click_rate', 'gmv_per_click'])

        if len(valid_vendors) > 0:
            log(f"  Vendors with valid efficiency metrics: {len(valid_vendors):,}", f)
            log("", f)

            # Quintile analysis: Do high-impression vendors have lower efficiency?
            valid_vendors['imp_quintile'] = pd.qcut(
                valid_vendors['total_impressions'],
                q=5,
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
            )

            quintile_stats = valid_vendors.groupby('imp_quintile', observed=True).agg({
                'VENDOR_ID': 'count',
                'total_impressions': 'mean',
                'click_rate': 'mean',
                'gmv_per_click': 'mean',
                'gmv_per_imp': 'mean'
            }).reset_index()

            log("  EFFICIENCY BY IMPRESSION QUINTILE:", f)
            log(f"  {'Quintile':<12} {'N':>8} {'Avg Imps':>12} {'CTR':>10} {'$/Click':>12} {'$/Imp':>12}", f)
            log(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*12}", f)

            for _, row in quintile_stats.iterrows():
                log(f"  {str(row['imp_quintile']):<12} {row['VENDOR_ID']:>8,} {row['total_impressions']:>12,.0f} {row['click_rate']*100:>9.3f}% ${row['gmv_per_click']:>10.2f} ${row['gmv_per_imp']:>10.4f}", f)

            log("", f)

            # Is there a pattern of decreasing efficiency with more impressions?
            q1_gmv_per_imp = quintile_stats[quintile_stats['imp_quintile'] == 'Q1 (Low)']['gmv_per_imp'].values
            q5_gmv_per_imp = quintile_stats[quintile_stats['imp_quintile'] == 'Q5 (High)']['gmv_per_imp'].values

            if len(q1_gmv_per_imp) > 0 and len(q5_gmv_per_imp) > 0:
                efficiency_ratio = q5_gmv_per_imp[0] / q1_gmv_per_imp[0] if q1_gmv_per_imp[0] > 0 else 0
                log(f"  Efficiency ratio (Q5/Q1): {efficiency_ratio:.3f}", f)

                if efficiency_ratio < 0.5:
                    log("    --> High-impression vendors are LESS efficient per impression", f)
                    log("    --> Suggests diminishing returns / potential cannibalization", f)
                elif efficiency_ratio > 1.5:
                    log("    --> High-impression vendors are MORE efficient per impression", f)
                    log("    --> Suggests economies of scale", f)
                else:
                    log("    --> Similar efficiency across impression levels", f)

                log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log("  CANNIBALIZATION INDICATORS:", f)
        log("", f)

        log("  1. STRONG negative correlation (organic_gmv vs impressions):", f)
        log("     --> Promoted impressions substitute for organic purchases", f)
        log("     --> Ads are 'stealing' credit from organic conversions", f)
        log("", f)

        log("  2. Diminishing returns (efficiency decreases with scale):", f)
        log("     --> Marginal impressions reach less-qualified users", f)
        log("     --> Platform may be showing ads to users who would buy anyway", f)
        log("", f)

        log("  3. To fully assess cannibalization:", f)
        log("     - Need total_gmv (all purchases, not just attributed)", f)
        log("     - Compare: total_gmv change vs promoted_gmv change", f)
        log("     - If 1:1 ratio, no cannibalization", f)
        log("     - If ratio < 1, some cannibalization exists", f)
        log("", f)

        log("  RECOMMENDATION:", f)
        log("    Run full cannibalization test (Script 08_cannibalization.py)", f)
        log("    with fresh Snowflake data to compute organic_gmv accurately.", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
