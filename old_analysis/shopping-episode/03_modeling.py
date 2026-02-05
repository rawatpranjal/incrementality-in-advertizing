#!/usr/bin/env python3
"""
03_modeling.py
Three analytical models for Shopping Episode incrementality.

RESEARCH HYPOTHESES:

Model I (Marketplace Elasticity):
- Does increasing ad clicks grow total GMV, or just cannibalize organic sales?
- β1 > 0 implies ads are additive; β1 ≈ 0 implies substitution

Model II (Vendor iROAS):
- Does a vendor paying for a click generate incremental revenue vs just bidding and losing?
- iROAS = β_clicks / AVG_CPC

Model III (Multinomial Logit):
- How does ad intensity shift the probability of choosing Vendor A vs Vendor B vs walking away?
- Explicitly models zero-sum competition
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "03_modeling.txt"


def log(msg: str, file=None):
    """Print and optionally write to file."""
    print(msg)
    if file:
        file.write(msg + "\n")


def build_marketplace_dataset(df_episodes: pd.DataFrame, df_events: pd.DataFrame,
                                df_purchases: pd.DataFrame, df_impressions: pd.DataFrame,
                                df_clicks: pd.DataFrame, f) -> pd.DataFrame:
    """Build episode-level dataset for Model I."""
    log("\n--- Building Marketplace Dataset (df_marketplace) ---", f)

    # Start with episodes
    df = df_episodes[['EPISODE_ID', 'USER_ID', 'START_TIME', 'END_TIME', 'DURATION_HOURS']].copy()

    # Map purchases to episodes
    df_events_pur = df_events[df_events['EVENT_TYPE'] == 'purchase']
    df_pur_ep = df_purchases.merge(
        df_events_pur[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        left_on=['USER_ID', 'PURCHASED_AT'],
        right_on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )
    gmv_per_ep = df_pur_ep.groupby('EPISODE_ID')['SPEND'].sum().reset_index()
    gmv_per_ep.columns = ['EPISODE_ID', 'TOTAL_GMV']

    df = df.merge(gmv_per_ep, on='EPISODE_ID', how='left')
    df['TOTAL_GMV'] = df['TOTAL_GMV'].fillna(0)

    # Map impressions to episodes
    if not df_impressions.empty:
        df_imp_ep = df_impressions.merge(
            df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )
        imp_per_ep = df_imp_ep.groupby('EPISODE_ID').size().reset_index(name='TOTAL_IMPRESSIONS')
        df = df.merge(imp_per_ep, on='EPISODE_ID', how='left')
        df['TOTAL_IMPRESSIONS'] = df['TOTAL_IMPRESSIONS'].fillna(0)
    else:
        df['TOTAL_IMPRESSIONS'] = 0

    # Map clicks to episodes
    if not df_clicks.empty:
        df_clk_ep = df_clicks.merge(
            df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )
        clk_per_ep = df_clk_ep.groupby('EPISODE_ID').size().reset_index(name='TOTAL_CLICKS')
        df = df.merge(clk_per_ep, on='EPISODE_ID', how='left')
        df['TOTAL_CLICKS'] = df['TOTAL_CLICKS'].fillna(0)
    else:
        df['TOTAL_CLICKS'] = 0

    # Derived features
    df['AD_INTENSITY'] = df['TOTAL_IMPRESSIONS'] / (df['DURATION_HOURS'] + 0.01)
    df['LOG_GMV'] = np.log(df['TOTAL_GMV'] + 1)
    df['LOG_CLICKS'] = np.log(df['TOTAL_CLICKS'] + 1)
    df['LOG_IMPRESSIONS'] = np.log(df['TOTAL_IMPRESSIONS'] + 1)
    df['LOG_DURATION'] = np.log(df['DURATION_HOURS'] + 1)
    df['HAS_PURCHASE'] = (df['TOTAL_GMV'] > 0).astype(int)

    log(f"  Episodes: {len(df):,}", f)
    log(f"  With purchases: {df['HAS_PURCHASE'].sum():,} ({df['HAS_PURCHASE'].mean()*100:.1f}%)", f)
    log(f"  Total GMV: ${df['TOTAL_GMV'].sum():,.2f}", f)
    log(f"  Total clicks: {df['TOTAL_CLICKS'].sum():,.0f}", f)
    log(f"  Total impressions: {df['TOTAL_IMPRESSIONS'].sum():,.0f}", f)

    df.to_parquet(DATA_DIR / "df_marketplace.parquet", index=False)
    log(f"  Saved: df_marketplace.parquet", f)

    return df


def build_vendor_dataset(df_episodes: pd.DataFrame, df_events: pd.DataFrame,
                          df_purchases: pd.DataFrame, df_impressions: pd.DataFrame,
                          df_clicks: pd.DataFrame, df_bids: pd.DataFrame, f) -> pd.DataFrame:
    """Build episode-vendor level dataset for Model II."""
    log("\n--- Building Vendor Dataset (df_vendor) ---", f)

    # Get all (episode, vendor) pairs from bids
    if df_bids.empty:
        log("  No bids data - cannot build vendor dataset", f)
        return pd.DataFrame()

    # Map bids to episodes via auctions
    df_auctions = pd.read_parquet(DATA_DIR / "auctions_users.parquet")
    df_bids_ep = df_bids.merge(
        df_auctions[['AUCTION_ID', 'USER_ID', 'CREATED_AT']],
        on='AUCTION_ID',
        how='left'
    )

    # Map to episodes
    df_bids_ep = df_bids_ep.merge(
        df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        left_on=['USER_ID', 'CREATED_AT'],
        right_on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )
    df_bids_ep = df_bids_ep.dropna(subset=['EPISODE_ID'])

    # Aggregate to (episode, vendor)
    vendor_bids = df_bids_ep.groupby(['EPISODE_ID', 'VENDOR_ID']).agg({
        'IS_WINNER': 'max',
        'AUCTION_ID': 'nunique',
        'RANKING': 'mean',
        'QUALITY': 'mean',
        'FINAL_BID': 'mean'
    }).reset_index()
    vendor_bids.columns = ['EPISODE_ID', 'VENDOR_ID', 'IS_WINNER', 'BID_COUNT', 'AVG_RANK', 'AVG_QUALITY', 'AVG_BID']

    # Get vendor clicks
    if not df_clicks.empty:
        df_clk_ep = df_clicks.merge(
            df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )
        vendor_clicks = df_clk_ep.groupby(['EPISODE_ID', 'VENDOR_ID']).size().reset_index(name='VENDOR_CLICKS')
        vendor_bids = vendor_bids.merge(vendor_clicks, on=['EPISODE_ID', 'VENDOR_ID'], how='left')
        vendor_bids['VENDOR_CLICKS'] = vendor_bids['VENDOR_CLICKS'].fillna(0)
    else:
        vendor_bids['VENDOR_CLICKS'] = 0

    # Get vendor GMV (purchases mapped to vendor)
    # This is tricky - we need to link purchases to vendors via impressions/clicks
    if not df_purchases.empty and not df_impressions.empty:
        # Map impressions to episodes
        df_imp_ep = df_impressions.merge(
            df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )

        # Map purchases to episodes
        df_events_pur = df_events[df_events['EVENT_TYPE'] == 'purchase']
        df_pur_ep = df_purchases.merge(
            df_events_pur[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
            left_on=['USER_ID', 'PURCHASED_AT'],
            right_on=['USER_ID', 'OCCURRED_AT'],
            how='left'
        )

        # Match purchases to vendors via product_id in impressions
        df_pur_vendor = df_pur_ep.merge(
            df_imp_ep[['EPISODE_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates(),
            on=['EPISODE_ID', 'PRODUCT_ID'],
            how='left'
        )

        vendor_gmv = df_pur_vendor.groupby(['EPISODE_ID', 'VENDOR_ID'])['SPEND'].sum().reset_index()
        vendor_gmv.columns = ['EPISODE_ID', 'VENDOR_ID', 'VENDOR_GMV']

        vendor_bids = vendor_bids.merge(vendor_gmv, on=['EPISODE_ID', 'VENDOR_ID'], how='left')
        vendor_bids['VENDOR_GMV'] = vendor_bids['VENDOR_GMV'].fillna(0)
    else:
        vendor_bids['VENDOR_GMV'] = 0

    # Add competitor clicks (total clicks in episode - vendor clicks)
    episode_clicks = vendor_bids.groupby('EPISODE_ID')['VENDOR_CLICKS'].sum().reset_index()
    episode_clicks.columns = ['EPISODE_ID', 'TOTAL_EPISODE_CLICKS']
    vendor_bids = vendor_bids.merge(episode_clicks, on='EPISODE_ID', how='left')
    vendor_bids['COMPETITOR_CLICKS'] = vendor_bids['TOTAL_EPISODE_CLICKS'] - vendor_bids['VENDOR_CLICKS']

    log(f"  (Episode, Vendor) pairs: {len(vendor_bids):,}", f)
    log(f"  Unique episodes: {vendor_bids['EPISODE_ID'].nunique():,}", f)
    log(f"  Unique vendors: {vendor_bids['VENDOR_ID'].nunique():,}", f)
    log(f"  With GMV > 0: {(vendor_bids['VENDOR_GMV'] > 0).sum():,}", f)
    log(f"  With clicks > 0: {(vendor_bids['VENDOR_CLICKS'] > 0).sum():,}", f)

    vendor_bids.to_parquet(DATA_DIR / "df_vendor.parquet", index=False)
    log(f"  Saved: df_vendor.parquet", f)

    return vendor_bids


def build_choice_dataset(df_episodes: pd.DataFrame, df_events: pd.DataFrame,
                          df_purchases: pd.DataFrame, df_impressions: pd.DataFrame, f) -> pd.DataFrame:
    """Build long-format choice dataset for Model III."""
    log("\n--- Building Choice Dataset (df_choice) ---", f)

    if df_impressions.empty:
        log("  No impressions data - cannot build choice dataset", f)
        return pd.DataFrame()

    # Map impressions to episodes
    df_imp_ep = df_impressions.merge(
        df_events[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )
    df_imp_ep = df_imp_ep.dropna(subset=['EPISODE_ID'])

    # Get top 2 vendors by impression count per episode
    vendor_sov = df_imp_ep.groupby(['EPISODE_ID', 'VENDOR_ID']).size().reset_index(name='IMP_COUNT')
    vendor_sov['SOV_RANK'] = vendor_sov.groupby('EPISODE_ID')['IMP_COUNT'].rank(method='first', ascending=False)

    focal = vendor_sov[vendor_sov['SOV_RANK'] == 1][['EPISODE_ID', 'VENDOR_ID', 'IMP_COUNT']].copy()
    focal.columns = ['EPISODE_ID', 'FOCAL_VENDOR', 'FOCAL_IMP']

    rival = vendor_sov[vendor_sov['SOV_RANK'] == 2][['EPISODE_ID', 'VENDOR_ID', 'IMP_COUNT']].copy()
    rival.columns = ['EPISODE_ID', 'RIVAL_VENDOR', 'RIVAL_IMP']

    # Map purchases to episodes
    df_events_pur = df_events[df_events['EVENT_TYPE'] == 'purchase']
    df_pur_ep = df_purchases.merge(
        df_events_pur[['USER_ID', 'OCCURRED_AT', 'EPISODE_ID']].drop_duplicates(),
        left_on=['USER_ID', 'PURCHASED_AT'],
        right_on=['USER_ID', 'OCCURRED_AT'],
        how='left'
    )

    # Match purchases to vendors
    df_pur_vendor = df_pur_ep.merge(
        df_imp_ep[['EPISODE_ID', 'PRODUCT_ID', 'VENDOR_ID']].drop_duplicates(),
        on=['EPISODE_ID', 'PRODUCT_ID'],
        how='left'
    )

    # Determine outcome per episode
    episode_outcome = df_episodes[['EPISODE_ID']].copy()
    episode_outcome = episode_outcome.merge(focal, on='EPISODE_ID', how='left')
    episode_outcome = episode_outcome.merge(rival, on='EPISODE_ID', how='left')

    # Check purchases
    ep_purchases = df_pur_vendor.groupby('EPISODE_ID').agg({
        'VENDOR_ID': 'first',
        'SPEND': 'sum'
    }).reset_index()
    ep_purchases.columns = ['EPISODE_ID', 'PURCHASED_VENDOR', 'GMV']

    episode_outcome = episode_outcome.merge(ep_purchases, on='EPISODE_ID', how='left')

    # Determine choice
    def get_choice(row):
        if pd.isna(row['PURCHASED_VENDOR']):
            return 'No_Buy'
        elif pd.isna(row['FOCAL_VENDOR']):
            return 'Organic'
        elif row['PURCHASED_VENDOR'] == row['FOCAL_VENDOR']:
            return 'Focal'
        elif row['PURCHASED_VENDOR'] == row.get('RIVAL_VENDOR'):
            return 'Rival'
        else:
            return 'Organic'

    episode_outcome['CHOICE'] = episode_outcome.apply(get_choice, axis=1)

    # Build long format (4 rows per episode: No_Buy, Organic, Focal, Rival)
    rows = []
    for _, ep in tqdm(episode_outcome.iterrows(), total=len(episode_outcome), desc="Building choice data"):
        ep_id = ep['EPISODE_ID']
        choice = ep['CHOICE']

        # No_Buy option
        rows.append({
            'EPISODE_ID': ep_id,
            'OPTION_ID': 'No_Buy',
            'IS_CHOSEN': 1 if choice == 'No_Buy' else 0,
            'HAS_ADS': 0,
            'IMP_COUNT': 0,
            'CLICK_COUNT': 0
        })

        # Organic option
        rows.append({
            'EPISODE_ID': ep_id,
            'OPTION_ID': 'Organic',
            'IS_CHOSEN': 1 if choice == 'Organic' else 0,
            'HAS_ADS': 0,
            'IMP_COUNT': 0,
            'CLICK_COUNT': 0
        })

        # Focal vendor option
        if pd.notna(ep['FOCAL_VENDOR']):
            rows.append({
                'EPISODE_ID': ep_id,
                'OPTION_ID': 'Focal',
                'IS_CHOSEN': 1 if choice == 'Focal' else 0,
                'HAS_ADS': 1,
                'IMP_COUNT': ep['FOCAL_IMP'],
                'CLICK_COUNT': 0  # Would need to join clicks
            })

        # Rival vendor option
        if pd.notna(ep['RIVAL_VENDOR']):
            rows.append({
                'EPISODE_ID': ep_id,
                'OPTION_ID': 'Rival',
                'IS_CHOSEN': 1 if choice == 'Rival' else 0,
                'HAS_ADS': 1,
                'IMP_COUNT': ep['RIVAL_IMP'],
                'CLICK_COUNT': 0
            })

    df_choice = pd.DataFrame(rows)

    log(f"  Episodes: {df_choice['EPISODE_ID'].nunique():,}", f)
    log(f"  Total rows: {len(df_choice):,}", f)
    log(f"  Choice distribution:", f)
    choice_dist = episode_outcome['CHOICE'].value_counts()
    for choice, count in choice_dist.items():
        log(f"    {choice}: {count:,} ({count/len(episode_outcome)*100:.1f}%)", f)

    df_choice.to_parquet(DATA_DIR / "df_choice.parquet", index=False)
    log(f"  Saved: df_choice.parquet", f)

    return df_choice


def run_model_i(df_mkt: pd.DataFrame, f):
    """Model I: Marketplace Elasticity (OLS)."""
    log("\n" + "=" * 80, f)
    log("MODEL I: MARKETPLACE ELASTICITY", f)
    log("=" * 80, f)

    log("\n--- Equation ---", f)
    log("log(GMV+1) = α + β₁·log(Clicks+1) + β₂·log(Duration+1) + ε", f)

    # Filter to valid observations
    df = df_mkt.dropna(subset=['LOG_GMV', 'LOG_CLICKS', 'LOG_DURATION'])

    if len(df) < 30:
        log(f"\n  ERROR: Only {len(df)} observations. Need at least 30.", f)
        return

    log(f"\n--- Data ---", f)
    log(f"  Observations: {len(df):,}", f)
    log(f"  GMV > 0: {(df['TOTAL_GMV'] > 0).sum():,} ({(df['TOTAL_GMV'] > 0).mean()*100:.1f}%)", f)

    # OLS regression
    X = df[['LOG_CLICKS', 'LOG_DURATION']]
    X = sm.add_constant(X)
    y = df['LOG_GMV']

    model = sm.OLS(y, X)
    results = model.fit()

    log("\n--- Results ---", f)
    log(results.summary().as_text(), f)

    # Interpretation
    log("\n--- Interpretation ---", f)
    beta_clicks = results.params.get('LOG_CLICKS', 0)
    p_clicks = results.pvalues.get('LOG_CLICKS', 1)

    log(f"  β_clicks = {beta_clicks:.4f} (p = {p_clicks:.4f})", f)
    if p_clicks < 0.05:
        if beta_clicks > 0.1:
            log("  → Ads are ADDITIVE: More clicks → More GMV", f)
        elif beta_clicks < -0.1:
            log("  → Ads are CANNIBALISTIC: More clicks → Less GMV", f)
        else:
            log("  → Ads are NEUTRAL: Clicks don't move GMV", f)
    else:
        log("  → Not statistically significant at α=0.05", f)


def run_model_ii(df_vendor: pd.DataFrame, f):
    """Model II: Vendor iROAS (Fixed Effects)."""
    log("\n" + "=" * 80, f)
    log("MODEL II: VENDOR iROAS", f)
    log("=" * 80, f)

    log("\n--- Equation ---", f)
    log("VENDOR_GMV = α + β₁·VENDOR_CLICKS + β₂·COMPETITOR_CLICKS + γ·BID_COUNT + μ_vendor + ε", f)

    if df_vendor.empty:
        log("\n  ERROR: No vendor data available", f)
        return

    # Filter to valid observations
    df = df_vendor.dropna(subset=['VENDOR_GMV', 'VENDOR_CLICKS', 'COMPETITOR_CLICKS', 'BID_COUNT'])

    if len(df) < 100:
        log(f"\n  ERROR: Only {len(df)} observations. Need at least 100.", f)
        return

    log(f"\n--- Data ---", f)
    log(f"  Observations: {len(df):,}", f)
    log(f"  Unique vendors: {df['VENDOR_ID'].nunique():,}", f)
    log(f"  With GMV > 0: {(df['VENDOR_GMV'] > 0).sum():,}", f)

    # OLS without FE first (for comparison)
    log("\n--- Model IIa: OLS (no FE) ---", f)
    X = df[['VENDOR_CLICKS', 'COMPETITOR_CLICKS', 'BID_COUNT']]
    X = sm.add_constant(X)
    y = df['VENDOR_GMV']

    model = sm.OLS(y, X)
    results = model.fit()

    log(f"  β_clicks = {results.params.get('VENDOR_CLICKS', 0):.4f} (SE = {results.bse.get('VENDOR_CLICKS', 0):.4f})", f)
    log(f"  β_competitor = {results.params.get('COMPETITOR_CLICKS', 0):.4f}", f)
    log(f"  R² = {results.rsquared:.4f}", f)

    # With Vendor FE (demeaning)
    log("\n--- Model IIb: Vendor Fixed Effects ---", f)
    try:
        import pyfixest as pf
        fe_model = pf.feols("VENDOR_GMV ~ VENDOR_CLICKS + COMPETITOR_CLICKS + BID_COUNT | VENDOR_ID",
                            data=df, vcov={'CRV1': 'VENDOR_ID'})
        log(fe_model.summary(), f)

        beta_clicks_fe = fe_model.coef().get('VENDOR_CLICKS', 0)
        log(f"\n  β_clicks (FE) = {beta_clicks_fe:.4f}", f)

        # iROAS calculation
        if 'AVG_BID' in df.columns:
            avg_cpc = df[df['VENDOR_CLICKS'] > 0]['AVG_BID'].mean()
            if avg_cpc > 0:
                iroas = beta_clicks_fe / avg_cpc
                log(f"  Avg CPC: ${avg_cpc:.2f}", f)
                log(f"  iROAS = β_clicks / CPC = {iroas:.2f}", f)
                if iroas > 1:
                    log("  → PROFITABLE: Each $1 spent generates ${:.2f}".format(iroas), f)
                else:
                    log("  → UNPROFITABLE: Each $1 spent generates ${:.2f}".format(iroas), f)

    except ImportError:
        log("  pyfixest not available - using manual demeaning", f)

        # Manual vendor demeaning
        for col in ['VENDOR_GMV', 'VENDOR_CLICKS', 'COMPETITOR_CLICKS', 'BID_COUNT']:
            df[f'{col}_dm'] = df.groupby('VENDOR_ID')[col].transform(lambda x: x - x.mean())

        X_dm = df[['VENDOR_CLICKS_dm', 'COMPETITOR_CLICKS_dm', 'BID_COUNT_dm']]
        X_dm = sm.add_constant(X_dm)
        y_dm = df['VENDOR_GMV_dm']

        model_dm = sm.OLS(y_dm, X_dm)
        results_dm = model_dm.fit()

        log(f"  β_clicks (FE) = {results_dm.params.get('VENDOR_CLICKS_dm', 0):.4f}", f)


def run_model_iii(df_choice: pd.DataFrame, f):
    """Model III: Multinomial Logit (Competition)."""
    log("\n" + "=" * 80, f)
    log("MODEL III: MULTINOMIAL LOGIT", f)
    log("=" * 80, f)

    log("\n--- Equation ---", f)
    log("P(Choice=j) = exp(αⱼ + β_imp·Iⱼ + β_ads·HAS_ADSⱼ) / Σexp(...)", f)
    log("Classes: No_Buy (base), Organic, Focal, Rival", f)

    if df_choice.empty:
        log("\n  ERROR: No choice data available", f)
        return

    # Pivot to wide format for MNLogit
    # Each episode needs features for each alternative
    episodes = df_choice['EPISODE_ID'].unique()

    if len(episodes) < 50:
        log(f"\n  ERROR: Only {len(episodes)} episodes. Need at least 50.", f)
        return

    log(f"\n--- Data ---", f)
    log(f"  Episodes: {len(episodes):,}", f)

    # Choice distribution
    chosen = df_choice[df_choice['IS_CHOSEN'] == 1]['OPTION_ID'].value_counts()
    log("  Choice distribution:", f)
    for opt, count in chosen.items():
        log(f"    {opt}: {count:,} ({count/len(episodes)*100:.1f}%)", f)

    # Prepare data for MNLogit
    # Create episode-level features with choice as outcome
    choice_map = {'No_Buy': 0, 'Organic': 1, 'Focal': 2, 'Rival': 3}
    episode_choice = df_choice[df_choice['IS_CHOSEN'] == 1].copy()
    episode_choice['CHOICE_CODE'] = episode_choice['OPTION_ID'].map(choice_map)

    # Get focal/rival impression counts
    focal_imp = df_choice[df_choice['OPTION_ID'] == 'Focal'].set_index('EPISODE_ID')['IMP_COUNT']
    rival_imp = df_choice[df_choice['OPTION_ID'] == 'Rival'].set_index('EPISODE_ID')['IMP_COUNT']

    episode_features = episode_choice[['EPISODE_ID', 'CHOICE_CODE']].copy()
    episode_features = episode_features.set_index('EPISODE_ID')
    episode_features['FOCAL_IMP'] = focal_imp
    episode_features['RIVAL_IMP'] = rival_imp
    episode_features = episode_features.fillna(0).reset_index()

    # Filter to episodes with all 4 options (complete cases)
    valid_episodes = df_choice.groupby('EPISODE_ID')['OPTION_ID'].nunique()
    valid_episodes = valid_episodes[valid_episodes >= 3].index
    episode_features = episode_features[episode_features['EPISODE_ID'].isin(valid_episodes)]

    if len(episode_features) < 50:
        log(f"\n  After filtering: Only {len(episode_features)} complete episodes", f)
        return

    log(f"  Complete episodes (>=3 options): {len(episode_features):,}", f)

    # Fit MNLogit
    X = episode_features[['FOCAL_IMP', 'RIVAL_IMP']]
    X = sm.add_constant(X)
    y = episode_features['CHOICE_CODE']

    try:
        model = MNLogit(y, X)
        results = model.fit(disp=0, maxiter=100)

        log("\n--- Results ---", f)
        log(results.summary().as_text(), f)

        log("\n--- Interpretation ---", f)
        log("  Base outcome: No_Buy (class 0)", f)
        log("  Coefficients show log-odds relative to No_Buy", f)

    except Exception as e:
        log(f"\n  ERROR fitting MNLogit: {e}", f)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("03_MODELING (Three Models)", f)
        log("=" * 80, f)
        log(f"\nTimestamp: {datetime.now()}", f)

        # Load data
        log("\n--- LOADING DATA ---", f)

        try:
            df_events = pd.read_parquet(DATA_DIR / "events.parquet")
            log(f"  events.parquet: {len(df_events):,} rows", f)
        except FileNotFoundError:
            log("  ERROR: events.parquet not found. Run 01_data_pull.py first.", f)
            return

        try:
            df_episodes = pd.read_parquet(DATA_DIR / "episodes.parquet")
            log(f"  episodes.parquet: {len(df_episodes):,} rows", f)
        except FileNotFoundError:
            log("  ERROR: episodes.parquet not found", f)
            return

        try:
            df_purchases = pd.read_parquet(DATA_DIR / "purchases.parquet")
            log(f"  purchases.parquet: {len(df_purchases):,} rows", f)
        except FileNotFoundError:
            df_purchases = pd.DataFrame()
            log("  purchases.parquet: Not found", f)

        try:
            df_impressions = pd.read_parquet(DATA_DIR / "impressions.parquet")
            log(f"  impressions.parquet: {len(df_impressions):,} rows", f)
        except FileNotFoundError:
            df_impressions = pd.DataFrame()
            log("  impressions.parquet: Not found", f)

        try:
            df_clicks = pd.read_parquet(DATA_DIR / "clicks.parquet")
            log(f"  clicks.parquet: {len(df_clicks):,} rows", f)
        except FileNotFoundError:
            df_clicks = pd.DataFrame()
            log("  clicks.parquet: Not found", f)

        try:
            df_bids = pd.read_parquet(DATA_DIR / "auctions_results.parquet")
            log(f"  auctions_results.parquet: {len(df_bids):,} rows", f)
        except FileNotFoundError:
            df_bids = pd.DataFrame()
            log("  auctions_results.parquet: Not found", f)

        # Build datasets
        log("\n" + "=" * 80, f)
        log("BUILDING ANALYSIS DATASETS", f)
        log("=" * 80, f)

        df_mkt = build_marketplace_dataset(df_episodes, df_events, df_purchases, df_impressions, df_clicks, f)
        df_vendor = build_vendor_dataset(df_episodes, df_events, df_purchases, df_impressions, df_clicks, df_bids, f)
        df_choice = build_choice_dataset(df_episodes, df_events, df_purchases, df_impressions, f)

        # Run models
        run_model_i(df_mkt, f)
        run_model_ii(df_vendor, f)
        run_model_iii(df_choice, f)

        log("\n" + "=" * 80, f)
        log("03_MODELING COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
