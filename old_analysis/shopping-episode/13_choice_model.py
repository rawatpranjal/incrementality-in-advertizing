#!/usr/bin/env python3
"""
13_choice_model.py
Event-level Multinomial Logit with 3 outcomes per purchase event.

Key design:
- Split sessions with multiple purchases into multiple choice events
- Event j uses exposures in window (t_{j-1}, t_j] where t_0 = session start
- For no-purchase sessions: one event with window = entire session

Outcomes per event (s,j):
  y_sj ∈ {0=no_purchase, 1=organic, 2=promoted}
  - 0: only for no-purchase sessions
  - 1: organic (purchased product NOT in promoted impressions in window)
  - 2: promoted (purchased product IN promoted impressions in window)

Model:
  V_sj,0 = 0 (base)
  V_sj,org = α_org + θ'x_sj + λ_week^org
  V_sj,prom = α_prom + θ'x_sj + β'w_sj + λ_week^prom

where w_sj = (clicks, impressions, avg_rank, median_price) in window (s,j)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.discrete.discrete_model import MNLogit
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "13_choice_model.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("13_CHOICE_MODEL (Event-Level Multinomial Logit)", f)
        log("=" * 80, f)
        log("", f)
        log("MODEL SPECIFICATION:", f)
        log("", f)
        log("Event-level 3-alternative MNL (one event per purchase, or one per no-purchase session):", f)
        log("  y_sj ∈ {0=no_purchase, 1=organic, 2=promoted}", f)
        log("", f)
        log("Event windows:", f)
        log("  - Session with purchases at t1 < t2 < ...: event j uses exposures in (t_{j-1}, t_j]", f)
        log("  - No-purchase session: one event with window = entire session", f)
        log("", f)
        log("Utilities (base = no_purchase):", f)
        log("  V_sj,0 = 0", f)
        log("  V_sj,org = α_org + θ'x_sj + λ_week^org", f)
        log("  V_sj,prom = α_prom + θ'x_sj + β'w_sj + λ_week^prom", f)
        log("", f)
        log("where:", f)
        log("  x_sj = event controls (event index, time since session start)", f)
        log("  w_sj = promoted exposure in window (clicks, impressions, avg_rank, median_price)", f)
        log("", f)
        log("Interpretation:", f)
        log("  β moves odds of promoted purchase vs {no purchase, organic}", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("LOADING DATA", f)
        log("-" * 40, f)

        # Load events with sessions
        log("\nLoading events with sessions...", f)
        events = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        log(f"  Events: {len(events):,}", f)

        # Load promoted events (clicks with auction metadata)
        log("\nLoading promoted events (clicks)...", f)
        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        log(f"  Promoted clicks: {len(promoted_events):,}", f)

        # Load impressions
        log("\nLoading impressions...", f)
        impressions = pd.read_parquet(SOURCE_DIR / 'impressions_365d.parquet')
        impressions['OCCURRED_AT'] = pd.to_datetime(impressions['OCCURRED_AT'])
        log(f"  Impressions: {len(impressions):,}", f)

        # Load all purchases
        log("\nLoading all purchases...", f)
        purchases_raw = pd.read_parquet(SOURCE_DIR / 'purchases_365d.parquet')
        purchases_raw['PURCHASED_AT'] = pd.to_datetime(purchases_raw['PURCHASED_AT'])
        log(f"  All purchases: {len(purchases_raw):,}", f)

        # Load catalog for prices
        log("\nLoading catalog...", f)
        catalog = pd.read_parquet(SOURCE_DIR / 'catalog_365d.parquet')
        price_lookup = catalog.set_index('PRODUCT_ID')['PRICE'].to_dict()
        log(f"  Catalog products: {len(catalog):,}", f)

        # ============================================================
        # 2. BUILD SESSION STRUCTURE
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING SESSION STRUCTURE", f)
        log("=" * 80, f)

        session_col = 'session_id_3d'

        # Get all sessions
        all_sessions = events[['user_id', session_col]].drop_duplicates()
        log(f"\nTotal unique sessions: {len(all_sessions):,}", f)

        # Get session start times
        session_starts = events.groupby(session_col)['timestamp'].min().to_dict()

        # Get click events
        click_events = events[events['event_type'] == 'click'].copy()
        log(f"Click events: {len(click_events):,}", f)

        # Get purchase events from events_with_sessions
        purchase_events = events[events['event_type'] == 'purchase'].copy()
        log(f"Purchase events (in sessions): {len(purchase_events):,}", f)

        # ============================================================
        # 3. MAP IMPRESSIONS TO SESSIONS WITH TIMESTAMPS
        # ============================================================
        log("", f)
        log("Mapping impressions to sessions...", f)

        # Get auction_id -> session mapping from promoted events
        click_auction_map = promoted_events[['user_id', 'auction_id', 'product_id']].drop_duplicates()

        # Merge clicks with session info to get auction -> session mapping
        clicks_with_sessions = click_events.merge(
            click_auction_map,
            on=['user_id', 'product_id'],
            how='inner'
        )
        log(f"  Clicks linked to auctions: {len(clicks_with_sessions):,}", f)

        # Get auction -> session mapping
        auction_session_map = clicks_with_sessions[['auction_id', session_col]].drop_duplicates()
        log(f"  Auctions mapped to sessions: {len(auction_session_map):,}", f)

        # Map impressions to sessions via auction_id
        impressions_with_sessions = impressions.merge(
            auction_session_map,
            left_on='AUCTION_ID',
            right_on='auction_id',
            how='inner'
        )
        log(f"  Impressions mapped to sessions: {len(impressions_with_sessions):,}", f)

        # Add prices to impressions
        impressions_with_sessions['price'] = impressions_with_sessions['PRODUCT_ID'].map(price_lookup)

        # ============================================================
        # 4. BUILD CHOICE EVENTS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING CHOICE EVENTS", f)
        log("=" * 80, f)

        log("\nCreating one event per purchase + one event per no-purchase session...", f)

        choice_events = []

        for idx, row in tqdm(all_sessions.iterrows(), total=len(all_sessions), desc="Building events"):
            session_id = row[session_col]
            user_id = row['user_id']

            session_start = session_starts.get(session_id)
            if session_start is None:
                continue

            # Get purchases in this session (sorted by time)
            session_purchases = purchase_events[purchase_events[session_col] == session_id].copy()
            session_purchases = session_purchases.sort_values('timestamp')

            # Get clicks in this session
            session_clicks = click_events[click_events[session_col] == session_id].copy()

            # Get impressions in this session
            session_impressions = impressions_with_sessions[
                impressions_with_sessions[session_col] == session_id
            ].copy()

            if len(session_purchases) == 0:
                # No-purchase session: one event with window = entire session
                # Get all exposures
                n_clicks = len(session_clicks)
                n_impressions = len(session_impressions)

                if n_impressions > 0:
                    median_price = session_impressions['price'].median()
                    min_price = session_impressions['price'].min()
                else:
                    median_price = np.nan
                    min_price = np.nan

                # Get week
                week = session_start.isocalendar()[1]

                # For no-purchase sessions, use session end as approximation
                session_events = events[events[session_col] == session_id]
                session_end = session_events['timestamp'].max()
                window_duration_sec = (session_end - session_start).total_seconds()

                choice_events.append({
                    'session_id': session_id,
                    'user_id': user_id,
                    'event_idx': 0,
                    'outcome': 0,  # no purchase
                    'n_clicks': n_clicks,
                    'n_impressions': n_impressions,
                    'median_price': median_price,
                    'min_price': min_price,
                    'week': week,
                    'window_start': session_start,
                    'window_end': session_end,
                    'window_duration_sec': window_duration_sec,
                    'purchased_product': None
                })

            else:
                # Session with purchases: one event per purchase
                purchase_times = session_purchases['timestamp'].tolist()
                purchase_products = session_purchases['product_id'].tolist()

                # Build windows: (session_start, t1], (t1, t2], ...
                window_starts = [session_start] + purchase_times[:-1]
                window_ends = purchase_times

                for j, (w_start, w_end, purchased_product) in enumerate(zip(window_starts, window_ends, purchase_products)):
                    # Get exposures in this window
                    window_clicks = session_clicks[
                        (session_clicks['timestamp'] > w_start) &
                        (session_clicks['timestamp'] <= w_end)
                    ]
                    window_impressions = session_impressions[
                        (session_impressions['OCCURRED_AT'] > w_start) &
                        (session_impressions['OCCURRED_AT'] <= w_end)
                    ]

                    n_clicks = len(window_clicks)
                    n_impressions = len(window_impressions)

                    if n_impressions > 0:
                        median_price = window_impressions['price'].median()
                        min_price = window_impressions['price'].min()
                        promoted_products = set(window_impressions['PRODUCT_ID'].unique())
                    else:
                        median_price = np.nan
                        min_price = np.nan
                        promoted_products = set()

                    # Determine outcome: promoted or organic?
                    if purchased_product in promoted_products:
                        outcome = 2  # promoted
                    else:
                        outcome = 1  # organic

                    # Get week
                    week = w_end.isocalendar()[1]

                    # Compute window duration
                    window_duration_sec = (w_end - w_start).total_seconds()

                    choice_events.append({
                        'session_id': session_id,
                        'user_id': user_id,
                        'event_idx': j,
                        'outcome': outcome,
                        'n_clicks': n_clicks,
                        'n_impressions': n_impressions,
                        'median_price': median_price,
                        'min_price': min_price,
                        'week': week,
                        'window_start': w_start,
                        'window_end': w_end,
                        'window_duration_sec': window_duration_sec,
                        'purchased_product': purchased_product
                    })

        events_df = pd.DataFrame(choice_events)
        log(f"\nTotal choice events: {len(events_df):,}", f)

        # ============================================================
        # 5. DESCRIPTIVE STATISTICS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("DESCRIPTIVE STATISTICS", f)
        log("=" * 80, f)

        log(f"\nEvent outcomes:", f)
        outcome_counts = events_df['outcome'].value_counts().sort_index()
        for outcome, count in outcome_counts.items():
            label = {0: 'no_purchase', 1: 'organic', 2: 'promoted'}[outcome]
            log(f"  {label}: {count:,} ({count/len(events_df)*100:.2f}%)", f)

        log(f"\nEvents per session:", f)
        events_per_session = events_df.groupby('session_id').size()
        log(f"  Mean: {events_per_session.mean():.2f}", f)
        log(f"  Max: {events_per_session.max()}", f)
        log(f"  Sessions with 1 event: {(events_per_session == 1).sum():,}", f)
        log(f"  Sessions with 2+ events: {(events_per_session >= 2).sum():,}", f)

        log(f"\n--- Exposure by Outcome ---", f)
        for outcome in [0, 1, 2]:
            label = {0: 'no_purchase', 1: 'organic', 2: 'promoted'}[outcome]
            subset = events_df[events_df['outcome'] == outcome]
            if len(subset) > 0:
                log(f"\n  {label} (n={len(subset):,}):", f)
                log(f"    Mean clicks in window: {subset['n_clicks'].mean():.3f}", f)
                log(f"    Mean impressions in window: {subset['n_impressions'].mean():.3f}", f)
                log(f"    Mean window duration (hours): {subset['window_duration_sec'].mean() / 3600:.2f}", f)
                log(f"    Median price: {subset['median_price'].median():.2f}", f)
                log(f"    Has any click: {(subset['n_clicks'] > 0).sum()} / {len(subset)} ({(subset['n_clicks'] > 0).mean()*100:.1f}%)", f)

        # ============================================================
        # 6. PREPARE DATA FOR MNLOGIT
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("PREPARING DATA FOR MNLOGIT", f)
        log("=" * 80, f)

        # Fill NAs
        events_df['n_clicks'] = events_df['n_clicks'].fillna(0)
        events_df['n_impressions'] = events_df['n_impressions'].fillna(0)
        events_df['median_price'] = events_df['median_price'].fillna(events_df['median_price'].median())
        events_df['min_price'] = events_df['min_price'].fillna(events_df['min_price'].median())
        events_df['window_duration_sec'] = events_df['window_duration_sec'].fillna(0)

        # Create binary indicators
        events_df['has_click'] = (events_df['n_clicks'] > 0).astype(int)
        events_df['has_impression'] = (events_df['n_impressions'] > 0).astype(int)

        # Create rate variables (per hour)
        # Avoid division by zero
        events_df['window_hours'] = events_df['window_duration_sec'] / 3600
        events_df['window_hours'] = events_df['window_hours'].replace(0, np.nan)
        events_df['clicks_per_hour'] = events_df['n_clicks'] / events_df['window_hours']
        events_df['imps_per_hour'] = events_df['n_impressions'] / events_df['window_hours']
        events_df['clicks_per_hour'] = events_df['clicks_per_hour'].fillna(0)
        events_df['imps_per_hour'] = events_df['imps_per_hour'].fillna(0)
        events_df['log_duration'] = np.log1p(events_df['window_duration_sec'])

        # Create event ID
        events_df['event_id'] = range(len(events_df))

        # Create week dummies
        events_df['week'] = events_df['week'].fillna(0).astype(int)
        week_dummies = pd.get_dummies(events_df['week'], prefix='week', drop_first=True)
        events_df = pd.concat([events_df, week_dummies], axis=1)

        log(f"\nFinal event dataset: {len(events_df):,} events", f)
        log(f"  Unique weeks: {events_df['week'].nunique()}", f)

        # --- DIAGNOSTIC TABLE ---
        log("\n--- Diagnostic Table (first 20 events) ---", f)
        diag_cols = ['event_id', 'outcome', 'has_click', 'n_clicks', 'n_impressions', 'window_duration_sec']
        diag_df = events_df[diag_cols].head(20)
        diag_df['window_hours'] = diag_df['window_duration_sec'] / 3600
        log(diag_df.to_string(index=False), f)

        log("\n--- Separation Check: has_click by outcome ---", f)
        for outcome in [0, 1, 2]:
            label = {0: 'no_purchase', 1: 'organic', 2: 'promoted'}[outcome]
            subset = events_df[events_df['outcome'] == outcome]
            n_with_click = (subset['has_click'] == 1).sum()
            n_without_click = (subset['has_click'] == 0).sum()
            log(f"  {label}: {n_with_click} with click, {n_without_click} without click", f)

        # ============================================================
        # 7. MULTINOMIAL LOGIT ESTIMATION
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MULTINOMIAL LOGIT ESTIMATION", f)
        log("=" * 80, f)

        if HAS_STATSMODELS:
            log("\nUsing statsmodels MNLogit", f)

            y = events_df['outcome']

            # ============================================================
            # MASTER MODEL: Clicks + Duration Control
            # ============================================================
            log("\n--- MASTER MODEL: Clicks + Duration Control ---", f)
            log("V_sj,0 = 0 (base)", f)
            log("V_sj,k = α_k + β_clicks * n_clicks + β_duration * log(duration)", f)
            log("", f)
            log("Rationale: Duration controls for time-at-risk confound", f)
            log("(Longer windows → more clicks AND more likely to purchase)", f)

            try:
                X = sm.add_constant(events_df[['n_clicks', 'log_duration']])
                model = MNLogit(y, X)
                result = model.fit(disp=0)
                log("\n" + str(result.summary()), f)

                params = result.params
                log(f"\n  Coefficients (base = no_purchase):", f)
                log(f"    Organic:  β_clicks={params.loc['n_clicks', 0]:.4f}, β_duration={params.loc['log_duration', 0]:.4f}", f)
                log(f"    Promoted: β_clicks={params.loc['n_clicks', 1]:.4f}, β_duration={params.loc['log_duration', 1]:.4f}", f)

            except Exception as e:
                log(f"Master model failed: {e}", f)

            # ============================================================
            # ROBUSTNESS: Clustered SEs by Session
            # ============================================================
            log("\n--- ROBUSTNESS: Session-Clustered Standard Errors ---", f)
            log("Multiple events per session share unobserved demand", f)

            try:
                result_cluster = model.fit(disp=0, cov_type='cluster', cov_kwds={'groups': events_df['session_id']})
                log("\n" + str(result_cluster.summary()), f)

                log(f"\n  Comparison: Clustered vs Non-clustered SEs", f)
                for var in ['n_clicks', 'log_duration']:
                    for outcome_idx, outcome_name in [(0, 'organic'), (1, 'promoted')]:
                        se_orig = result.bse.loc[var, outcome_idx]
                        se_clust = result_cluster.bse.loc[var, outcome_idx]
                        ratio = se_clust / se_orig if se_orig > 0 else np.nan
                        log(f"    {outcome_name} {var}: SE {se_orig:.4f} → {se_clust:.4f} (ratio={ratio:.2f})", f)

            except Exception as e:
                log(f"Clustered SEs failed: {e}", f)
                log("  Falling back to robust (HC0) SEs...", f)
                try:
                    result_robust = model.fit(disp=0, cov_type='HC0')
                    log("\n" + str(result_robust.summary()), f)
                except Exception as e2:
                    log(f"  Robust SEs also failed: {e2}", f)

        else:
            log("\nstatsmodels not available for MNLogit", f)

        # ============================================================
        # 8. CONTINGENCY ANALYSIS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("CONTINGENCY ANALYSIS", f)
        log("=" * 80, f)

        log("\n--- Cross-tabulation: Has Click × Outcome ---", f)
        ct = pd.crosstab(events_df['has_click'], events_df['outcome'], margins=True)
        ct.columns = ['no_purchase', 'organic', 'promoted', 'Total']
        ct.index = ['no_click', 'has_click', 'Total']
        log(str(ct), f)

        log("\n--- Row percentages (within click status) ---", f)
        ct_pct = pd.crosstab(events_df['has_click'], events_df['outcome'], normalize='index') * 100
        ct_pct.columns = ['no_purchase', 'organic', 'promoted']
        ct_pct.index = ['no_click', 'has_click']
        log(str(ct_pct.round(2)), f)

        # Chi-square test
        from scipy.stats import chi2_contingency
        ct_raw = pd.crosstab(events_df['has_click'], events_df['outcome'])
        if ct_raw.shape == (2, 3):
            chi2, p_chi2, dof, expected = chi2_contingency(ct_raw)
            log(f"\n  Chi-square test: χ² = {chi2:.2f}, df = {dof}, p = {p_chi2:.6f}", f)

        # ============================================================
        # 9. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SUMMARY", f)
        log("=" * 80, f)

        log(f"\nEvent-level sample:", f)
        log(f"  Total events: {len(events_df):,}", f)
        log(f"  Unique sessions: {events_df['session_id'].nunique():,}", f)
        for outcome in [0, 1, 2]:
            label = {0: 'no_purchase', 1: 'organic', 2: 'promoted'}[outcome]
            n = (events_df['outcome'] == outcome).sum()
            pct = n / len(events_df) * 100
            log(f"  {label}: {n:,} ({pct:.2f}%)", f)

        log(f"\nExposure by outcome:", f)
        for outcome in [0, 1, 2]:
            label = {0: 'no_purchase', 1: 'organic', 2: 'promoted'}[outcome]
            subset = events_df[events_df['outcome'] == outcome]
            if len(subset) > 0:
                click_rate = (subset['n_clicks'] > 0).mean() * 100
                log(f"  {label}: {click_rate:.2f}% had clicks in window", f)

        log("\n" + "=" * 80, f)
        log("13_CHOICE_MODEL COMPLETE", f)
        log("=" * 80, f)

        # Save data
        events_df.to_parquet(DATA_DIR / 'event_choice_data.parquet', index=False)
        log(f"\nEvent data saved to: {DATA_DIR / 'event_choice_data.parquet'}", f)

if __name__ == "__main__":
    main()
