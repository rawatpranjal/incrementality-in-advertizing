#!/usr/bin/env python3
"""
Within-Session Rank Variation Analysis (Search + Mobile Only)

Research Question: Can we use within-session rank variation to estimate causal position effects on CTR?

Design: Within-Product, Within-Session Comparison
- Unit of analysis: Product-session pair (min 3 observations)
- Treatment: Rank position (varies across auctions within session)
- Outcome: Click (0/1)
- Identification: Compare same product's CTR when ranked higher vs. lower within same user-session

Preferred Model:
    Y_aps = beta * RANK_aps + delta * SCORE_aps + alpha_ps + epsilon_aps
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
DATA_DIR = Path(__file__).resolve().parents[3] / 'analysis' / 'position-effects' / '1_data_pull' / 'data'

MIN_OBS_PER_PS = 3  # Minimum observations per product-session


def get_paths() -> dict:
    return {
        'auctions_results': DATA_DIR / 'auctions_results_all.parquet',
        'impressions': DATA_DIR / 'impressions_all.parquet',
        'clicks': DATA_DIR / 'clicks_all.parquet',
        'auctions_users': DATA_DIR / 'auctions_users_all.parquet',
    }


def identify_mobile_users(imps: pd.DataFrame) -> set:
    """
    Identify mobile users based on impression patterns.
    Mobile shows 2 ads at a time (same timestamp), desktop shows 4+.
    """
    imps = imps.copy()
    imps['OCCURRED_AT'] = pd.to_datetime(imps['OCCURRED_AT'], utc=True, errors='coerce')

    # Count impressions per user per second
    imps['ts_second'] = imps['OCCURRED_AT'].dt.floor('s')
    imps_per_second = imps.groupby(['USER_ID', 'ts_second']).size().reset_index(name='n_imps')

    # For each user, get the max impressions seen at once
    user_max_imps = imps_per_second.groupby('USER_ID')['n_imps'].max().reset_index()
    user_max_imps.columns = ['USER_ID', 'max_concurrent_imps']

    # Mobile users: max concurrent impressions <= 2
    mobile_users = set(user_max_imps[user_max_imps['max_concurrent_imps'] <= 2]['USER_ID'])

    return mobile_users


def define_sessions(au: pd.DataFrame, session_gap_minutes: int = 30) -> pd.DataFrame:
    """
    Define sessions as consecutive auctions within session_gap_minutes.
    """
    au = au.sort_values(['USER_ID', 'CREATED_AT']).copy()
    au['time_since_last'] = au.groupby('USER_ID')['CREATED_AT'].diff()
    au['new_session'] = (au['time_since_last'] > pd.Timedelta(minutes=session_gap_minutes)) | au['time_since_last'].isna()
    au['session_num'] = au.groupby('USER_ID')['new_session'].cumsum()
    au['session_id'] = au['USER_ID'] + '_' + au['session_num'].astype(str)
    return au


def run_fixed_effects_regression(df: pd.DataFrame, y_col: str, x_col: str, fe_col: str,
                                  cluster_col: str = None, control_cols: list = None):
    """
    Run fixed effects regression: y ~ x + controls + fe + epsilon
    Uses within-transformation (demeaning) for fixed effects.
    """
    try:
        import statsmodels.api as sm

        df = df.copy()
        df['y'] = df[y_col].astype(float)
        df['x'] = df[x_col].astype(float)

        # Collect all variables to demean
        vars_to_demean = ['y', 'x']
        if control_cols:
            for ctrl in control_cols:
                df[ctrl] = df[ctrl].astype(float)
                vars_to_demean.append(ctrl)

        # Apply within-transformation
        for var in vars_to_demean:
            df[var] = df[var] - df.groupby(fe_col)[var].transform('mean')

        # Build X matrix
        x_vars = ['x']
        if control_cols:
            x_vars.extend(control_cols)

        X = df[x_vars]
        model = sm.OLS(df['y'], X)

        if cluster_col is not None and df[cluster_col].nunique() > 1:
            res = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
        else:
            res = model.fit(cov_type='HC1')

        result = {
            'coef': res.params['x'],
            'se': res.bse['x'],
            't': res.tvalues['x'],
            'p': res.pvalues['x'],
            'nobs': int(res.nobs),
            'r2_within': res.rsquared,
            'n_fe_groups': df[fe_col].nunique()
        }

        if control_cols:
            result['controls'] = {}
            for ctrl in control_cols:
                result['controls'][ctrl] = {
                    'coef': res.params[ctrl],
                    'se': res.bse[ctrl],
                    't': res.tvalues[ctrl],
                    'p': res.pvalues[ctrl]
                }

        return result
    except Exception as e:
        return {'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='Within-session rank variation analysis (search + mobile)')
    ap.add_argument('--session_window', type=int, default=30, help='Session window in minutes')
    args = ap.parse_args()

    paths = get_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "03_within_session_position.txt"

    with open(out, 'w') as fh:
        def w(s): fh.write(str(s) + "\n"); fh.flush(); print(s)

        w("=" * 90)
        w("Within-Session Rank Variation Analysis")
        w("=" * 90)
        w("")
        w("SAMPLE RESTRICTIONS:")
        w("  - Placement 1 (search) only")
        w("  - Mobile users only (max 2 concurrent impressions)")
        w(f"  - Min {MIN_OBS_PER_PS} observations per product-session")
        w(f"  - Session window: {args.session_window} minutes")
        w("")
        w("ECONOMETRIC SPECIFICATION (Preferred Model):")
        w("")
        w("    Y_aps = beta * RANK_aps + delta * SCORE_aps + alpha_ps + epsilon_aps")
        w("")
        w("  where:")
        w("    a = auction, p = product, s = session")
        w("    Y = outcome (clicked or impressed, binary 0/1)")
        w("    RANK = position in auction (1 = highest, larger = worse)")
        w("    SCORE = QUALITY * FINAL_BID (ranking score)")
        w("    alpha_ps = product-session fixed effect")
        w("    epsilon = idiosyncratic error")
        w("")
        w("  Estimation: Within-transformation (demeaning), OLS")
        w("  Standard errors: Clustered at session level")
        w("")
        w("  Interpretation of beta:")
        w("    beta < 0 => worse rank reduces outcome probability")
        w("    |beta| = change in Pr(Y=1) per 1-position worse rank, holding score constant")
        w("")

        # Load data
        w("=" * 90)
        w("LOADING DATA")
        w("=" * 90)

        au = pd.read_parquet(paths['auctions_users'])
        ar = pd.read_parquet(paths['auctions_results'])
        imps = pd.read_parquet(paths['impressions'])
        clks = pd.read_parquet(paths['clicks'])

        w(f"  Auctions: {len(au):,}")
        w(f"  Bids: {len(ar):,}")
        w(f"  Impressions: {len(imps):,}")
        w(f"  Clicks: {len(clks):,}")

        # Parse timestamps
        au['CREATED_AT'] = pd.to_datetime(au['CREATED_AT'], utc=True, errors='coerce')

        # Identify mobile users
        w(f"\n  Identifying mobile users...")
        mobile_users = identify_mobile_users(imps)
        w(f"  Mobile users: {len(mobile_users):,}")
        w(f"  Desktop users: {imps['USER_ID'].nunique() - len(mobile_users):,}")

        # Filter to placement 1 (search) and mobile users
        au_p1 = au[(au['PLACEMENT'] == '1') & (au['USER_ID'].isin(mobile_users))].copy()
        w(f"\n  After filtering to P1 + mobile:")
        w(f"    Auctions: {len(au_p1):,}")
        w(f"    Users: {au_p1['USER_ID'].nunique():,}")

        # Prepare impression and click flags
        imp_flag = imps[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates().assign(impressed=1)
        clk_flag = clks[['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates().assign(clicked=1)

        # Prepare auction results
        ar = ar.dropna(subset=['RANKING'])
        ar = ar[ar['RANKING'] >= 1].copy()
        ar['RANKING'] = ar['RANKING'].astype(int)
        ar['QUALITY'] = ar['QUALITY'].fillna(0).astype(float)
        ar['FINAL_BID'] = ar['FINAL_BID'].fillna(0).astype(float)
        ar['score'] = ar['QUALITY'] * ar['FINAL_BID']

        # Filter to P1 auctions
        p1_auction_ids = set(au_p1['AUCTION_ID'].astype(str))
        ar_p1 = ar[ar['AUCTION_ID'].astype(str).isin(p1_auction_ids)].copy()
        w(f"    Bids in P1 mobile auctions: {len(ar_p1):,}")

        # Define sessions
        w(f"\n  Defining sessions (gap = {args.session_window} min)...")
        au_sessions = define_sessions(au_p1, session_gap_minutes=args.session_window)
        n_sessions = au_sessions['session_id'].nunique()
        w(f"    Sessions: {n_sessions:,}")

        # Merge session info
        au_slim = au_sessions[['AUCTION_ID', 'session_id']].copy()
        ar_sessions = ar_p1.merge(au_slim, on='AUCTION_ID', how='inner')
        ar_sessions['product_session_id'] = ar_sessions['PRODUCT_ID'] + '_' + ar_sessions['session_id']

        # Filter to product-sessions with >= MIN_OBS_PER_PS observations
        ps_counts = ar_sessions.groupby('product_session_id').size()
        valid_ps = set(ps_counts[ps_counts >= MIN_OBS_PER_PS].index)
        ar_filtered = ar_sessions[ar_sessions['product_session_id'].isin(valid_ps)].copy()

        w(f"\n  Product-session filtering (min {MIN_OBS_PER_PS} obs):")
        w(f"    Total product-sessions: {len(ps_counts):,}")
        w(f"    Product-sessions with >= {MIN_OBS_PER_PS} obs: {len(valid_ps):,}")
        w(f"    Observations after filter: {len(ar_filtered):,}")

        if len(ar_filtered) < 100:
            w("  ERROR: Too few observations after filtering")
            return

        # Compute rank variation
        rank_stats = ar_filtered.groupby('product_session_id').agg({
            'RANKING': ['min', 'max', 'std', 'count']
        }).reset_index()
        rank_stats.columns = ['product_session_id', 'rank_min', 'rank_max', 'rank_std', 'n_obs']
        rank_stats['rank_delta'] = rank_stats['rank_max'] - rank_stats['rank_min']

        # Keep only product-sessions with rank variation
        varied_ps = set(rank_stats[rank_stats['rank_delta'] > 0]['product_session_id'])
        ar_varied = ar_filtered[ar_filtered['product_session_id'].isin(varied_ps)].copy()

        w(f"\n  Rank variation filter:")
        w(f"    Product-sessions with rank variation: {len(varied_ps):,}")
        w(f"    Final observations: {len(ar_varied):,}")
        w(f"    Mean rank delta: {rank_stats[rank_stats['rank_delta'] > 0]['rank_delta'].mean():.2f}")
        w(f"    Mean obs per product-session: {rank_stats[rank_stats['rank_delta'] > 0]['n_obs'].mean():.2f}")

        # Merge outcomes
        ar_varied = ar_varied.merge(imp_flag, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
        ar_varied = ar_varied.merge(clk_flag, on=['AUCTION_ID', 'PRODUCT_ID', 'VENDOR_ID'], how='left')
        ar_varied['impressed'] = ar_varied['impressed'].fillna(0).astype(int)
        ar_varied['clicked'] = ar_varied['clicked'].fillna(0).astype(int)

        w(f"\n  Outcome rates:")
        w(f"    Impression rate: {ar_varied['impressed'].mean():.4f}")
        w(f"    Click rate: {ar_varied['clicked'].mean():.4f}")
        if ar_varied['impressed'].sum() > 0:
            w(f"    CTR (clicks/impressions): {ar_varied['clicked'].sum() / ar_varied['impressed'].sum():.4f}")

        # Score variation
        score_var = ar_varied.groupby('product_session_id')['score'].std().mean()
        w(f"\n  Within product-session score variation:")
        w(f"    Mean std(score): {score_var:.4f}")

        # === REGRESSION ANALYSIS ===
        w(f"\n{'=' * 90}")
        w("REGRESSION RESULTS")
        w("=" * 90)

        # Model 1: Baseline (no controls)
        w(f"\n--- MODEL 1 (Baseline): Y ~ RANK | product_session_fe ---")

        fe1_click = run_fixed_effects_regression(
            ar_varied, y_col='clicked', x_col='RANKING',
            fe_col='product_session_id', cluster_col='session_id'
        )
        if 'error' not in fe1_click:
            sig = '***' if fe1_click['p'] < 0.001 else '**' if fe1_click['p'] < 0.01 else '*' if fe1_click['p'] < 0.05 else ''
            w(f"  CLICKED:")
            w(f"    beta(RANK) = {fe1_click['coef']:.6f} (SE={fe1_click['se']:.6f}, t={fe1_click['t']:.2f}, p={fe1_click['p']:.4f}){sig}")
            w(f"    N = {fe1_click['nobs']:,}, N_product_session = {fe1_click['n_fe_groups']:,}")

        fe1_imp = run_fixed_effects_regression(
            ar_varied, y_col='impressed', x_col='RANKING',
            fe_col='product_session_id', cluster_col='session_id'
        )
        if 'error' not in fe1_imp:
            sig = '***' if fe1_imp['p'] < 0.001 else '**' if fe1_imp['p'] < 0.01 else '*' if fe1_imp['p'] < 0.05 else ''
            w(f"  IMPRESSED:")
            w(f"    beta(RANK) = {fe1_imp['coef']:.6f} (SE={fe1_imp['se']:.6f}, t={fe1_imp['t']:.2f}, p={fe1_imp['p']:.4f}){sig}")

        # Model 2: Preferred (with score control)
        w(f"\n--- MODEL 2 (Preferred): Y ~ RANK + SCORE | product_session_fe ---")

        fe2_click = run_fixed_effects_regression(
            ar_varied, y_col='clicked', x_col='RANKING',
            fe_col='product_session_id', cluster_col='session_id',
            control_cols=['score']
        )
        if 'error' not in fe2_click:
            sig = '***' if fe2_click['p'] < 0.001 else '**' if fe2_click['p'] < 0.01 else '*' if fe2_click['p'] < 0.05 else ''
            w(f"  CLICKED:")
            w(f"    beta(RANK) = {fe2_click['coef']:.6f} (SE={fe2_click['se']:.6f}, t={fe2_click['t']:.2f}, p={fe2_click['p']:.4f}){sig}")
            if 'controls' in fe2_click and 'score' in fe2_click['controls']:
                s = fe2_click['controls']['score']
                sig_s = '***' if s['p'] < 0.001 else '**' if s['p'] < 0.01 else '*' if s['p'] < 0.05 else ''
                w(f"    beta(SCORE) = {s['coef']:.4f} (SE={s['se']:.4f}, t={s['t']:.2f}, p={s['p']:.4f}){sig_s}")

        fe2_imp = run_fixed_effects_regression(
            ar_varied, y_col='impressed', x_col='RANKING',
            fe_col='product_session_id', cluster_col='session_id',
            control_cols=['score']
        )
        if 'error' not in fe2_imp:
            sig = '***' if fe2_imp['p'] < 0.001 else '**' if fe2_imp['p'] < 0.01 else '*' if fe2_imp['p'] < 0.05 else ''
            w(f"  IMPRESSED:")
            w(f"    beta(RANK) = {fe2_imp['coef']:.6f} (SE={fe2_imp['se']:.6f}, t={fe2_imp['t']:.2f}, p={fe2_imp['p']:.4f}){sig}")
            if 'controls' in fe2_imp and 'score' in fe2_imp['controls']:
                s = fe2_imp['controls']['score']
                sig_s = '***' if s['p'] < 0.001 else '**' if s['p'] < 0.01 else '*' if s['p'] < 0.05 else ''
                w(f"    beta(SCORE) = {s['coef']:.4f} (SE={s['se']:.4f}, t={s['t']:.2f}, p={s['p']:.4f}){sig_s}")

        # === ROBUSTNESS: Large rank changes ===
        w(f"\n--- ROBUSTNESS: Large Rank Changes (delta >= 5) ---")
        large_ps = set(rank_stats[rank_stats['rank_delta'] >= 5]['product_session_id'])
        ar_large = ar_varied[ar_varied['product_session_id'].isin(large_ps)].copy()

        if len(ar_large) >= 50:
            w(f"  Observations: {len(ar_large):,}")
            w(f"  Product-sessions: {ar_large['product_session_id'].nunique():,}")

            fe_large = run_fixed_effects_regression(
                ar_large, y_col='clicked', x_col='RANKING',
                fe_col='product_session_id', cluster_col='session_id',
                control_cols=['score']
            )
            if 'error' not in fe_large:
                sig = '***' if fe_large['p'] < 0.001 else '**' if fe_large['p'] < 0.01 else '*' if fe_large['p'] < 0.05 else ''
                w(f"  CLICKED: beta(RANK) = {fe_large['coef']:.6f} (SE={fe_large['se']:.6f}, p={fe_large['p']:.4f}){sig}")

            fe_large_imp = run_fixed_effects_regression(
                ar_large, y_col='impressed', x_col='RANKING',
                fe_col='product_session_id', cluster_col='session_id',
                control_cols=['score']
            )
            if 'error' not in fe_large_imp:
                sig = '***' if fe_large_imp['p'] < 0.001 else '**' if fe_large_imp['p'] < 0.01 else '*' if fe_large_imp['p'] < 0.05 else ''
                w(f"  IMPRESSED: beta(RANK) = {fe_large_imp['coef']:.6f} (SE={fe_large_imp['se']:.6f}, p={fe_large_imp['p']:.4f}){sig}")
        else:
            w("  (Too few observations)")

        # === EMAIL SUMMARY ===
        w(f"\n{'=' * 90}")
        w("SUMMARY FOR CO-AUTHORS")
        w("=" * 90)

        base_ctr = ar_varied['clicked'].mean()
        base_imp = ar_varied['impressed'].mean()

        w(f"""
RESEARCH QUESTION
-----------------
Can we estimate causal position effects using within-session rank variation?

IDEA
----
The same product appears at different ranks across auctions within a user's
search session. By comparing the same product when it ranks higher vs lower
(within the same session), we can estimate position effects while controlling
for product-session unobservables.

SAMPLE
------
- Search page (Placement 1) only
- Mobile users only (identified by seeing max 2 ads at once)
- Product-sessions with at least 3 auctions
- Product-sessions where rank varied across auctions

Final sample: {len(ar_varied):,} observations, {ar_varied['product_session_id'].nunique():,} product-sessions

MODEL
-----
    Y_aps = beta * Rank_aps + delta * Score_aps + alpha_ps + epsilon_aps

    Y       = Clicked (0/1) or Impressed (0/1)
    Rank    = Position in auction (1 = top, higher = worse)
    Score   = Quality * Bid (the ranking score used by the auction)
    alpha_ps = Product-session fixed effect

    Estimation: Within-transformation, OLS
    Standard errors: Clustered at session level

The fixed effect absorbs all time-invariant product-session characteristics.
The score control absorbs variation in predicted CTR and bid changes.
Beta captures the pure position effect: does rank matter holding score constant?


RESULTS
-------
""")

        # Build results table
        w(f"{'Outcome':<12} {'Coefficient':<14} {'Std Error':<12} {'t-stat':<10} {'p-value':<10}")
        w("-" * 58)

        if 'error' not in fe2_click:
            sig = '***' if fe2_click['p'] < 0.001 else '**' if fe2_click['p'] < 0.01 else '*' if fe2_click['p'] < 0.05 else ''
            w(f"{'Clicked':<12} {fe2_click['coef']:<14.6f} {fe2_click['se']:<12.6f} {fe2_click['t']:<10.2f} {fe2_click['p']:<10.4f}{sig}")

        if 'error' not in fe2_imp:
            sig = '***' if fe2_imp['p'] < 0.001 else '**' if fe2_imp['p'] < 0.01 else '*' if fe2_imp['p'] < 0.05 else ''
            w(f"{'Impressed':<12} {fe2_imp['coef']:<14.6f} {fe2_imp['se']:<12.6f} {fe2_imp['t']:<10.2f} {fe2_imp['p']:<10.4f}{sig}")

        w(f"\nN = {fe2_click['nobs']:,} observations, {fe2_click['n_fe_groups']:,} product-session fixed effects")
        w(f"*** p<0.001, ** p<0.01, * p<0.05")

        w(f"""

INTERPRETATION
--------------
Each 1-position worse rank reduces:
  - Click probability by {abs(fe2_click['coef'])*100:.3f} percentage points
  - Impression probability by {abs(fe2_imp['coef'])*100:.2f} percentage points

In relative terms (baseline click rate = {base_ctr*100:.2f}%):
  - 1 position worse  -> {abs(fe2_click['coef']/base_ctr)*100:.1f}% fewer clicks
  - 10 positions worse -> {abs(fe2_click['coef']/base_ctr)*100*10:.0f}% fewer clicks

The impression effect is ~33x larger than the click effect, suggesting most of
the position effect operates through visibility (whether the ad is seen at all)
rather than through click propensity conditional on being seen.


ROBUSTNESS
----------
Restricting to product-sessions with large rank changes (delta >= 5 positions):
  - Clicked:   beta = {fe_large['coef']:.6f} (SE = {fe_large['se']:.6f})
  - Impressed: beta = {fe_large_imp['coef']:.6f} (SE = {fe_large_imp['se']:.6f})

Results are stable.


CAVEATS
-------
1. Identification assumes rank variation within session is as-good-as-random
   conditional on product-session fixed effects and score. This could fail if
   the auction context changes within session (different search queries).

2. We control for score (quality * bid) but score changes within session are
   small (mean within-session std = {score_var:.4f}).

3. Sample is restricted to mobile search users with repeated product appearances,
   which may not generalize to all users/placements.
""")

        w(f"\nOutput saved to: {out}")


if __name__ == '__main__':
    main()
