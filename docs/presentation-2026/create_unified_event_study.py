#!/usr/bin/env python3
"""
Create unified event study figure for presentation.
Shows Impressions, Clicks, GMV on same figure with standardized scaling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NumPy 2.0 compatibility
np.NaN = np.nan

from differences import ATTgt

def load_and_prep_data():
    """Load panel data."""
    path = Path(__file__).parent.parent / 'staggered-adoption-final' / 'data' / 'panel_total_gmv.parquet'
    panel = pd.read_parquet(path)

    first_period = pd.to_datetime('2025-03-24')
    panel['period'] = ((pd.to_datetime(panel['week']) - first_period).dt.days // 7 + 1).astype(int)
    panel['cohort_period'] = panel['cohort'].apply(
        lambda x: np.nan if pd.isna(x) else int((pd.to_datetime(x) - first_period).days // 7 + 1)
    )
    panel['entity'] = panel['VENDOR_ID'].astype('category').cat.codes

    # Create conversion indicator (binary: any purchase)
    panel['conversion'] = (panel['total_gmv'] > 0).astype(int)

    # Log GMV (with log(1+x) transform for zeros)
    panel['log_gmv'] = np.log1p(panel['total_gmv'])

    df = panel[['entity', 'period', 'cohort_period', 'impressions', 'clicks', 'conversion', 'total_gmv', 'log_gmv']].copy()
    df = df.rename(columns={'cohort_period': 'cohort'})
    df = df.set_index(['entity', 'period'])

    return df

def get_event_study_data(df, outcome):
    """Run CS estimation and return event study data."""
    att_gt = ATTgt(data=df, cohort_name='cohort')
    att_gt.fit(formula=outcome, control_group='never_treated', est_method='reg', n_jobs=-1, progress_bar=False)

    event = att_gt.aggregate('event')
    event.columns = ['_'.join(filter(None, map(str, col))).strip() for col in event.columns]
    event = event.reset_index()

    return event

def create_unified_figure(events_dict, save_path):
    """
    Create unified event study figure with 3 panels (standardized to % of baseline).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    outcomes = ['impressions', 'clicks', 'total_gmv']
    titles = ['Impressions', 'Clicks', 'Total GMV']
    colors = ['#2166AC', '#4393C3', '#762A83']

    # Baseline means for standardization
    baselines = {'impressions': 1.32, 'clicks': 0.04, 'total_gmv': 1.81}

    for idx, (outcome, title, color) in enumerate(zip(outcomes, titles, colors)):
        ax = axes[idx]
        event = events_dict[outcome].copy()
        event = event.sort_values('relative_period')

        # Filter to -8 to 15
        mask = (event['relative_period'] >= -8) & (event['relative_period'] <= 15)
        event = event[mask]

        e = event['relative_period'].values
        theta = event['EventAggregation_ATT'].values
        lower = event['EventAggregation_pointwise conf. band_lower'].values
        upper = event['EventAggregation_pointwise conf. band_upper'].values

        # Standardize to % of baseline
        baseline = baselines[outcome]
        theta_pct = (theta / baseline) * 100
        lower_pct = (lower / baseline) * 100
        upper_pct = (upper / baseline) * 100

        # Plot
        ax.fill_between(e, lower_pct, upper_pct, alpha=0.2, color=color)
        ax.plot(e, theta_pct, 'o-', color=color, markersize=4, linewidth=1.5)

        # Reference lines
        ax.axvline(x=0, color='#B2182B', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvspan(-8, 0, alpha=0.05, color='gray')

        # Labels
        ax.set_xlabel('Weeks Since Treatment', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Effect (% of Baseline)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(-8.5, 15.5)

        # Add significance indicator
        post_mean = theta_pct[(e >= 0) & (e <= 10)].mean()
        if outcome == 'total_gmv':
            ax.annotate('n.s.', xy=(10, post_mean), fontsize=11, color='gray', fontweight='bold')
        else:
            ax.annotate('***', xy=(10, post_mean), fontsize=11, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_4panel_figure(events_dict, save_path):
    """
    Create unified event study figure with 3 panels: Impressions, Clicks, Log GMV.
    For impressions/clicks: % of baseline. For log GMV: raw coefficient (approx % change).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    outcomes = ['impressions', 'clicks', 'log_gmv']
    titles = ['Impressions', 'Clicks', 'Log GMV']
    colors = ['#2166AC', '#4393C3', '#762A83']

    # Baseline means for standardization (only for impressions/clicks)
    baselines = {'impressions': 1.32, 'clicks': 0.04}

    for idx, (outcome, title, color) in enumerate(zip(outcomes, titles, colors)):
        ax = axes[idx]
        event = events_dict[outcome].copy()
        event = event.sort_values('relative_period')

        # Filter to -8 to 15
        mask = (event['relative_period'] >= -8) & (event['relative_period'] <= 15)
        event = event[mask]

        e = event['relative_period'].values
        theta = event['EventAggregation_ATT'].values
        lower = event['EventAggregation_pointwise conf. band_lower'].values
        upper = event['EventAggregation_pointwise conf. band_upper'].values

        if outcome == 'log_gmv':
            # For log GMV: show raw coefficients * 100 (approx % change)
            theta_plot = theta * 100
            lower_plot = lower * 100
            upper_plot = upper * 100
            ylabel = 'Effect (% Change)'
        else:
            # Standardize to % of baseline
            baseline = baselines[outcome]
            theta_plot = (theta / baseline) * 100
            lower_plot = (lower / baseline) * 100
            upper_plot = (upper / baseline) * 100
            ylabel = 'Effect (% of Baseline)'

        # Plot
        ax.fill_between(e, lower_plot, upper_plot, alpha=0.2, color=color)
        ax.plot(e, theta_plot, 'o-', color=color, markersize=4, linewidth=1.5)

        # Reference lines
        ax.axvline(x=0, color='#B2182B', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvspan(-8, 0, alpha=0.05, color='gray')

        # Labels
        ax.set_xlabel('Weeks Since Treatment', fontsize=10)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(-8.5, 15.5)

        # Add significance indicator
        post_mean = theta_plot[(e >= 0) & (e <= 10)].mean()
        if outcome == 'log_gmv':
            ax.annotate('n.s.', xy=(10, post_mean), fontsize=11, color='gray', fontweight='bold')
        else:
            ax.annotate('***', xy=(10, post_mean), fontsize=11, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def create_stacked_figure(events_dict, save_path):
    """
    Alternative: Single panel with all three outcomes overlaid (normalized to effect size).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    outcomes = ['impressions', 'clicks', 'total_gmv']
    labels = ['Impressions (+80%***)', 'Clicks (+80%***)', 'GMV (+14%, n.s.)']
    colors = ['#2166AC', '#4393C3', '#B2182B']
    markers = ['o', 's', '^']

    # Baseline means
    baselines = {'impressions': 1.32, 'clicks': 0.04, 'total_gmv': 1.81}

    for outcome, label, color, marker in zip(outcomes, labels, colors, markers):
        event = events_dict[outcome].copy()
        event = event.sort_values('relative_period')

        mask = (event['relative_period'] >= -6) & (event['relative_period'] <= 12)
        event = event[mask]

        e = event['relative_period'].values
        theta = event['EventAggregation_ATT'].values

        # Normalize to % of baseline
        baseline = baselines[outcome]
        theta_pct = (theta / baseline) * 100

        ax.plot(e, theta_pct, marker=marker, color=color, markersize=5,
                linewidth=1.5, label=label, alpha=0.85)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, label='Treatment')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvspan(-6, 0, alpha=0.08, color='gray')

    ax.set_xlabel('Weeks Relative to Ad Adoption', fontsize=11)
    ax.set_ylabel('Effect (% of Baseline Mean)', fontsize=11)
    ax.set_title('Event Study: Causal Effect of Advertising', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6.5, 12.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    from tqdm import tqdm

    print("Loading data...")
    df = load_and_prep_data()

    print("Running event study estimations...")
    events_dict = {}
    for outcome in tqdm(['impressions', 'clicks', 'conversion', 'total_gmv', 'log_gmv']):
        print(f"  Estimating: {outcome}")
        events_dict[outcome] = get_event_study_data(df, outcome)

    # Output directory
    out_dir = Path(__file__).parent / 'tabfig'
    out_dir.mkdir(exist_ok=True)

    print("\nCreating figures...")
    create_unified_figure(events_dict, out_dir / 'event_study_unified_3panel.png')
    create_4panel_figure(events_dict, out_dir / 'event_study_unified_4panel.png')
    create_stacked_figure(events_dict, out_dir / 'event_study_unified_overlay.png')

    # Print summary of log_gmv results
    print("\n=== Log GMV Event Study Results ===")
    log_event = events_dict['log_gmv'].copy()
    log_event = log_event.sort_values('relative_period')
    mask = (log_event['relative_period'] >= -3) & (log_event['relative_period'] <= 10)
    log_event = log_event[mask]
    print(log_event[['relative_period', 'EventAggregation_ATT',
                     'EventAggregation_pointwise conf. band_lower',
                     'EventAggregation_pointwise conf. band_upper']].to_string(index=False))

    print("\nDone!")
