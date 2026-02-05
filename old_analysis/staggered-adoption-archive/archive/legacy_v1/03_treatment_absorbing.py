#!/usr/bin/env python3
"""
EDA Q3: How "Absorbing" is the Treatment?

Of vendors who switch Promoted Closet ON (Spend > 0 in week t), what percentage
switch it OFF (Spend = 0) in week t+1? If this "flicker rate" is high (>20%),
the standard Callaway-Sant'Anna estimator (which assumes treatment is irreversible)
is invalid, and we must use a reversible treatment estimator.
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
OUTPUT_FILE = RESULTS_DIR / "03_treatment_absorbing.txt"

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
        log("EDA Q3: TREATMENT ABSORBING STATE", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  Of vendors who switch ON (Spend > 0), what % switch OFF next week?", f)
        log("  Threshold: If flicker rate > 20%, CS irreversibility assumption violated.", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Load vendor×week panel with spend indicator", f)
        log("  2. For each vendor, track ON/OFF transitions week-to-week", f)
        log("  3. Compute flicker rate = P(OFF at t+1 | ON at t)", f)
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
        log(f"  Unique weeks: {panel['week'].nunique()}", f)
        log("", f)

        # Ensure sorted by vendor and week
        panel = panel.sort_values(['VENDOR_ID', 'week']).reset_index(drop=True)

        # -----------------------------------------------------------------
        # Create treatment indicator
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("TREATMENT TRANSITIONS ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Treatment = has_spend (binary indicator for positive spend)
        if 'has_spend' not in panel.columns:
            panel['has_spend'] = (panel['total_spend'] > 0).astype(int)

        # Create lagged treatment status
        panel['has_spend_prev'] = panel.groupby('VENDOR_ID')['has_spend'].shift(1)
        panel['has_spend_next'] = panel.groupby('VENDOR_ID')['has_spend'].shift(-1)

        # Drop rows without valid transitions
        transitions = panel.dropna(subset=['has_spend_prev', 'has_spend_next'])

        log(f"  Total transitions (vendor-weeks with prev & next): {len(transitions):,}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Transition matrix
        # -----------------------------------------------------------------
        log("TRANSITION MATRIX:", f)
        log("-" * 40, f)
        log("", f)

        # Count transitions: (prev_state, current_state)
        transition_counts = transitions.groupby(['has_spend_prev', 'has_spend']).size().unstack(fill_value=0)

        log("  State at t-1 (rows) vs State at t (cols):", f)
        log("", f)
        log(f"              | OFF (t)  | ON (t)   |", f)
        log(f"  ------------|----------|----------|", f)

        for prev_state in [0, 1]:
            state_label = "OFF (t-1)" if prev_state == 0 else "ON (t-1) "
            off_count = transition_counts.loc[prev_state, 0] if 0 in transition_counts.columns else 0
            on_count = transition_counts.loc[prev_state, 1] if 1 in transition_counts.columns else 0
            log(f"  {state_label} | {off_count:>8,} | {on_count:>8,} |", f)

        log("", f)

        # Transition probabilities
        log("TRANSITION PROBABILITIES:", f)
        log("-" * 40, f)
        log("", f)

        # P(ON | was OFF) = "Turn ON" rate
        off_to_on = ((transitions['has_spend_prev'] == 0) & (transitions['has_spend'] == 1)).sum()
        off_total = (transitions['has_spend_prev'] == 0).sum()
        turn_on_rate = off_to_on / off_total if off_total > 0 else 0

        # P(OFF | was ON) = "Flicker" rate (treatment reversal)
        on_to_off = ((transitions['has_spend_prev'] == 1) & (transitions['has_spend'] == 0)).sum()
        on_total = (transitions['has_spend_prev'] == 1).sum()
        flicker_rate = on_to_off / on_total if on_total > 0 else 0

        # P(ON | was ON) = "Persistence" rate
        on_to_on = ((transitions['has_spend_prev'] == 1) & (transitions['has_spend'] == 1)).sum()
        persistence_rate = on_to_on / on_total if on_total > 0 else 0

        log(f"  P(Turn ON | was OFF):  {turn_on_rate*100:.2f}%", f)
        log(f"  P(Stay OFF | was OFF): {(1-turn_on_rate)*100:.2f}%", f)
        log("", f)
        log(f"  P(Stay ON | was ON):   {persistence_rate*100:.2f}% (persistence)", f)
        log(f"  P(Turn OFF | was ON):  {flicker_rate*100:.2f}% (FLICKER RATE)", f)
        log("", f)

        # -----------------------------------------------------------------
        # Flicker analysis by vendor
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("VENDOR-LEVEL FLICKER ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # For each vendor, count ON→OFF transitions
        vendor_flickers = transitions[transitions['has_spend_prev'] == 1].groupby('VENDOR_ID').agg({
            'has_spend': ['sum', 'count']  # sum = stayed ON, count = total ON periods
        }).reset_index()
        vendor_flickers.columns = ['VENDOR_ID', 'stayed_on', 'on_periods']
        vendor_flickers['flicker_count'] = vendor_flickers['on_periods'] - vendor_flickers['stayed_on']
        vendor_flickers['vendor_flicker_rate'] = vendor_flickers['flicker_count'] / vendor_flickers['on_periods']

        log(f"  Vendors with at least 1 ON period: {len(vendor_flickers):,}", f)
        log("", f)

        # Distribution of vendor-level flicker rates
        log("  VENDOR-LEVEL FLICKER RATE DISTRIBUTION:", f)
        never_flicker = (vendor_flickers['vendor_flicker_rate'] == 0).sum()
        always_flicker = (vendor_flickers['vendor_flicker_rate'] == 1).sum()
        log(f"    Vendors who NEVER flicker (rate=0): {never_flicker:,} ({never_flicker/len(vendor_flickers)*100:.1f}%)", f)
        log(f"    Vendors who ALWAYS flicker (rate=1): {always_flicker:,} ({always_flicker/len(vendor_flickers)*100:.1f}%)", f)
        log("", f)

        quantiles = vendor_flickers['vendor_flicker_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        log("  Quantiles:", f)
        for q, val in quantiles.items():
            log(f"    P{int(q*100):02d}: {val*100:.1f}% flicker rate", f)
        log("", f)

        # -----------------------------------------------------------------
        # Multiple flicker events per vendor
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("MULTI-FLICKER VENDORS", f)
        log("-" * 40, f)
        log("", f)

        flicker_count_dist = vendor_flickers['flicker_count'].value_counts().sort_index()
        log("  Number of ON→OFF transitions per vendor:", f)
        for n_flickers, count in flicker_count_dist.head(10).items():
            log(f"    {n_flickers} flickers: {count:,} vendors", f)
        if len(flicker_count_dist) > 10:
            log(f"    ... (max flickers: {vendor_flickers['flicker_count'].max()})", f)
        log("", f)

        # -----------------------------------------------------------------
        # Time pattern of flickers
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("FLICKER TIMING PATTERN", f)
        log("-" * 40, f)
        log("", f)

        # Flicker rate by week
        weekly_flickers = transitions[transitions['has_spend_prev'] == 1].copy()
        weekly_flickers['flicker'] = (weekly_flickers['has_spend'] == 0).astype(int)
        weekly_rates = weekly_flickers.groupby('week').agg({
            'flicker': ['sum', 'count']
        }).reset_index()
        weekly_rates.columns = ['week', 'flicker_count', 'on_count']
        weekly_rates['flicker_rate'] = weekly_rates['flicker_count'] / weekly_rates['on_count']

        log(f"  {'Week':<12} {'ON→OFF':>10} {'Total ON':>12} {'Flicker Rate':>14}", f)
        log(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*14}", f)
        for _, row in weekly_rates.iterrows():
            week_str = str(row['week'])[:10]
            log(f"  {week_str:<12} {row['flicker_count']:>10,} {row['on_count']:>12,} {row['flicker_rate']*100:>13.1f}%", f)
        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log(f"  OVERALL FLICKER RATE: {flicker_rate*100:.2f}%", f)
        log("", f)

        if flicker_rate > 0.20:
            log("  [WARNING] Flicker rate exceeds 20%.", f)
            log("  The standard Callaway-Sant'Anna estimator assumes irreversible treatment.", f)
            log("  RECOMMENDATIONS:", f)
            log("    1. Use a reversible treatment estimator (de Chaisemartin & D'Haultfoeuille)", f)
            log("    2. Or, define treatment as 'ever adopted' to make it absorbing", f)
            log("    3. Or, restrict to vendors who never revert", f)
        elif flicker_rate > 0.10:
            log("  [CAUTION] Flicker rate is 10-20%.", f)
            log("  Some treatment reversals observed. Consider sensitivity analysis.", f)
        else:
            log("  [OK] Flicker rate is low (<10%).", f)
            log("  The absorbing state assumption is reasonable for most vendors.", f)

        log("", f)

        # CS assumption validity
        absorbing_vendors = (vendor_flickers['vendor_flicker_rate'] == 0).sum()
        absorbing_pct = absorbing_vendors / len(vendor_flickers) * 100
        log(f"  Vendors with perfectly absorbing treatment: {absorbing_vendors:,} ({absorbing_pct:.1f}%)", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


if __name__ == "__main__":
    main()
