"""
Run FE regressions on the rebuilt 28,619 row panel.
Replicates results from archive/results.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("WARNING: pyfixest not installed. Using statsmodels.")

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

DATA_DIR = Path('/Users/pranjal/Code/marketplace-incrementality/shopping-episode/archive/data')

print("=" * 80)
print("RUNNING FE REGRESSIONS")
print("=" * 80)

# Load panel
panel = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
print(f"\nPanel loaded: {len(panel):,} rows")
print(f"Users: {panel['user_id'].nunique():,}")
print(f"Weeks: {panel['year_week'].nunique()}")
print(f"Vendors: {panel['vendor_id'].nunique():,}")

# ============================================================================
# FE PROGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("FE PROGRESSION: How β changes with fixed effects")
print("=" * 80)

results = []

def get_nobs(m):
    """Get number of observations from pyfixest model."""
    try:
        return m.nobs
    except AttributeError:
        try:
            return m._N
        except AttributeError:
            return len(m._Y)

if HAS_PYFIXEST:
    # 1. Pooled OLS (no FE)
    m = pf.feols("Y ~ C", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('None (Pooled OLS)', m.coef()['C'], m.se()['C'], m.pvalue()['C'], len(panel)))

    # 2. Week only
    m = pf.feols("Y ~ C | year_week", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('Week only', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 3. User only
    m = pf.feols("Y ~ C | user_id", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('User only', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 4. Vendor only
    m = pf.feols("Y ~ C | vendor_id", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('Vendor only', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 5. User + Week
    m = pf.feols("Y ~ C | user_id + year_week", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('User + Week', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 6. User + Vendor
    m = pf.feols("Y ~ C | user_id + vendor_id", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('User + Vendor', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 7. Week + Vendor
    m = pf.feols("Y ~ C | year_week + vendor_id", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('Week + Vendor', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

    # 8. Full: User + Week + Vendor
    m = pf.feols("Y ~ C | user_id + year_week + vendor_id", data=panel, vcov={'CRV1': 'user_id'})
    results.append(('User + Week + Vendor', m.coef()['C'], m.se()['C'], m.pvalue()['C'], get_nobs(m)))

else:
    # Fallback to statsmodels (slower, no clustering)
    from statsmodels.regression.linear_model import OLS

    # 1. Pooled OLS
    X = sm.add_constant(panel[['C']])
    m = OLS(panel['Y'], X).fit()
    results.append(('None (Pooled OLS)', m.params['C'], m.bse['C'], m.pvalues['C'], len(panel)))

    print("Note: Using statsmodels without clustering. Install pyfixest for proper FE models.")

# Print results
print("\n| Specification | β(C) | SE | p-value | N |")
print("|---------------|------|-----|---------|---|")
for name, beta, se, pval, n in results:
    print(f"| {name} | {beta:.4f} | {se:.4f} | {pval:.4f} | {n:,} |")

# ============================================================================
# KEY FINDING
# ============================================================================
print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)

if len(results) >= 8:
    ols_beta = results[0][1]
    full_beta = results[7][1]
    print(f"\nOLS (no FE):        β = {ols_beta:+.4f}")
    print(f"Full FE (U+W+V):    β = {full_beta:+.4f}")
    print(f"\nSign flip: {'+' if ols_beta > 0 else '-'} → {'+' if full_beta > 0 else '-'}")
    print("\nInterpretation: Adding Vendor FE flips the sign from positive to negative.")
    print("This suggests selection bias is absorbed by fixed effects.")

# ============================================================================
# SESSION-BASED MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SESSION-BASED MODELS (Model 3)")
print("=" * 80)

events = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')

session_results = []
for gap_days in [1, 2, 3, 5, 7]:
    session_col = f'session_id_{gap_days}d'

    # Build session-week-vendor panel
    clicks_stv = events[events['event_type'] == 'click'].groupby(
        [session_col, 'year_week', 'vendor_id']
    ).size().reset_index(name='C')

    spend_stv = events[events['event_type'] == 'purchase'].groupby(
        [session_col, 'year_week', 'vendor_id']
    )['spend'].sum().reset_index(name='Y')

    panel_stv = clicks_stv.merge(spend_stv, on=[session_col, 'year_week', 'vendor_id'], how='outer')
    panel_stv['C'] = panel_stv['C'].fillna(0)
    panel_stv['Y'] = panel_stv['Y'].fillna(0)

    n_sessions = panel_stv[session_col].nunique()

    if HAS_PYFIXEST:
        try:
            m = pf.feols(f"Y ~ C | {session_col} + year_week + vendor_id", data=panel_stv, vcov={'CRV1': session_col})
            session_results.append((gap_days, len(panel_stv), n_sessions, m.coef()['C'], m.se()['C'], m.pvalue()['C']))
        except Exception as e:
            session_results.append((gap_days, len(panel_stv), n_sessions, np.nan, np.nan, np.nan))
    else:
        session_results.append((gap_days, len(panel_stv), n_sessions, np.nan, np.nan, np.nan))

print("\n| Gap | N | Sessions | β | SE | p |")
print("|-----|---|----------|---|----|----|")
for gap, n, sess, beta, se, pval in session_results:
    if np.isnan(beta):
        print(f"| {gap}d | {n:,} | {sess:,} | -- | -- | -- |")
    else:
        print(f"| {gap}d | {n:,} | {sess:,} | {beta:.4f} | {se:.4f} | {pval:.4f} |")

# ============================================================================
# CONVERSION MODEL
# ============================================================================
print("\n" + "=" * 80)
print("CONVERSION MODEL (Binary D = 1{Y > 0})")
print("=" * 80)

panel['D'] = (panel['Y'] > 0).astype(int)
print(f"Conversion rate: {panel['D'].mean()*100:.2f}%")

if HAS_PYFIXEST:
    try:
        m = pf.feols("D ~ C | user_id + year_week + vendor_id", data=panel, vcov={'CRV1': 'user_id'})
        print(f"\nD ~ C | user + week + vendor")
        print(f"  β = {m.coef()['C']:.6f}")
        print(f"  SE = {m.se()['C']:.6f}")
        print(f"  p = {m.pvalue()['C']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Key Results:
1. OLS (no FE): β = +0.22 (p=0.08) - positive association
2. Full FE (U+W+V): β = -0.23 (p=0.21) - negative, not significant

Interpretation:
- The positive OLS coefficient reflects selection: users who click more are interested buyers
- Adding Vendor FE flips the sign: within vendor, more clicks = comparison shopping
- The null result (p=0.21) suggests no causal effect of clicks on spend after controls

The 575 Problem:
- Only 575 user-vendor pairs have click variation for identification
- 99.2% of cells have zero spend (extreme sparsity)
- Limited power to detect effects
""")

print("\nDone.")
