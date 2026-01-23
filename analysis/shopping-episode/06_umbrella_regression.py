#!/usr/bin/env python3
"""
06_umbrella_regression.py
Estimates comprehensive model nesting multiple mechanism tests:
- Short-run activation: β_0·C_stv
- Awareness/adstock: Σ β_ℓ·C_{s,t-ℓ,v} for ℓ=1,2,3,4
- View-through: η_0·I_stv + Σ η_ℓ·I_{s,t-ℓ,v}
- Search-cost/friction: δ_1·(C_stv·Short_st)
- Position bias: δ_2·(C_stv·Top_stv)
- Competition/substitution: ρ·C^{(-v)}_st
- Anchoring: κ·A_st

Fixed Effects: User (α_u) + Week (λ_t) + Vendor (φ_v)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "06_umbrella_regression.txt"

GAP_DAYS = 3  # Primary session gap
MAX_LAG = 4

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("06_UMBRELLA_REGRESSION", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script tests whether clicking on one vendor affects spend on other vendors.", f)
        log("Cross-vendor spillovers would indicate that sponsored ads generate platform-wide", f)
        log("awareness or shopping intent, not just vendor-specific effects. We test for", f)
        log("competition/substitution (ρ < 0 means clicks on other vendors reduce own vendor", f)
        log("spend) versus complementarity (ρ > 0 means clicks on other vendors increase own", f)
        log("vendor spend). Lagged click effects test whether clicking today affects spending", f)
        log("in future weeks, capturing delayed attribution or habit formation. The anchoring", f)
        log("hypothesis tests whether initial exposure price affects subsequent spend. The", f)
        log("friction hypothesis tests whether clicks are more effective in short sessions.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("Loading data...", f)

        panel_stv = pd.read_parquet(DATA_DIR / f'panel_stv_{GAP_DAYS}d.parquet')
        panel_utv = pd.read_parquet(DATA_DIR / 'panel_utv.parquet')
        promoted_events = pd.read_parquet(DATA_DIR / 'promoted_events.parquet')
        events_with_sessions = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')

        log(f"Panel (s,t,v) {GAP_DAYS}d: {len(panel_stv):,} rows", f)
        log(f"Panel (u,t,v): {len(panel_utv):,} rows", f)
        log(f"Promoted events: {len(promoted_events):,} rows", f)
        log(f"Events with sessions: {len(events_with_sessions):,} rows", f)

        # ============================================================
        # 2. BUILD UMBRELLA VARIABLES
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING UMBRELLA VARIABLES", f)
        log("=" * 80, f)

        df = panel_stv.copy()

        weeks_ordered = sorted(df['year_week'].unique())
        week_to_idx = {w: i for i, w in enumerate(weeks_ordered)}
        df['week_idx'] = df['year_week'].map(week_to_idx)

        log(f"Weeks available: {len(weeks_ordered)}", f)
        log(f"Sessions: {df['session_id'].nunique():,}", f)
        log(f"Users: {df['user_id'].nunique():,}", f)

        # 2.1 Lagged clicks
        log("\n--- Building Lagged Clicks ---", f)
        click_by_week = df[['session_id', 'vendor_id', 'week_idx', 'C']].copy()

        for lag in tqdm(range(1, MAX_LAG + 1), desc="Building click lags"):
            lagged = click_by_week.copy()
            lagged['week_idx'] = lagged['week_idx'] + lag
            lagged = lagged.rename(columns={'C': f'C_lag{lag}'})

            df = df.merge(
                lagged[['session_id', 'vendor_id', 'week_idx', f'C_lag{lag}']],
                on=['session_id', 'vendor_id', 'week_idx'],
                how='left'
            )
            df[f'C_lag{lag}'] = df[f'C_lag{lag}'].fillna(0)

        for lag in range(1, MAX_LAG + 1):
            log(f"C_lag{lag}: mean = {df[f'C_lag{lag}'].mean():.3f}", f)

        # 2.2 Impressions
        log("\n--- Building Impressions ---", f)
        if 'I' in panel_utv.columns:
            impressions_utv = panel_utv[['user_id', 'year_week', 'vendor_id', 'I']].copy()
            df = df.merge(
                impressions_utv,
                on=['user_id', 'year_week', 'vendor_id'],
                how='left'
            )
            df['I'] = df['I'].fillna(0)
            log(f"Impressions (I): mean = {df['I'].mean():.2f}", f)

            impressions_utv['week_idx'] = impressions_utv['year_week'].map(week_to_idx)
            for lag in tqdm(range(1, MAX_LAG + 1), desc="Building impression lags"):
                lagged = impressions_utv.copy()
                lagged['week_idx'] = lagged['week_idx'] + lag
                lagged = lagged.rename(columns={'I': f'I_lag{lag}'})
                df = df.merge(
                    lagged[['user_id', 'vendor_id', 'week_idx', f'I_lag{lag}']],
                    on=['user_id', 'vendor_id', 'week_idx'],
                    how='left'
                )
                df[f'I_lag{lag}'] = df[f'I_lag{lag}'].fillna(0)
        else:
            log("Impressions not available - skipping", f)
            df['I'] = 0
            for lag in range(1, MAX_LAG + 1):
                df[f'I_lag{lag}'] = 0

        # 2.3 Friction proxy
        log("\n--- Building Friction Proxy (Short_st) ---", f)
        session_col = f'session_id_{GAP_DAYS}d'
        session_stats = events_with_sessions.groupby(session_col).agg({
            'product_id': 'count',
        }).reset_index()
        session_stats.columns = ['session_id', 'n_events']
        median_events = session_stats['n_events'].median()
        session_stats['is_short'] = (session_stats['n_events'] < median_events).astype(int)
        df = df.merge(session_stats[['session_id', 'n_events', 'is_short']], on='session_id', how='left')
        df['is_short'] = df['is_short'].fillna(0)
        log(f"Median session events: {median_events}", f)
        log(f"Short sessions: {df['is_short'].mean()*100:.1f}%", f)

        # 2.4 Position proxy
        log("\n--- Building Position Proxy (Top_stv) ---", f)
        promoted_events['click_time'] = pd.to_datetime(promoted_events['click_time'])
        promoted_events['year_week'] = (
            promoted_events['click_time'].dt.year.astype(str) + '_W' +
            promoted_events['click_time'].dt.isocalendar().week.astype(str).str.zfill(2)
        )
        rank_by_utv = promoted_events.groupby(['user_id', 'year_week', 'vendor_id']).agg({
            'ranking': 'mean',
            'is_winner': 'mean'
        }).reset_index()
        rank_by_utv.columns = ['user_id', 'year_week', 'vendor_id', 'avg_rank', 'share_rank1']
        df = df.merge(rank_by_utv, on=['user_id', 'year_week', 'vendor_id'], how='left')
        df['share_rank1'] = df['share_rank1'].fillna(0)
        df['avg_rank'] = df['avg_rank'].fillna(0)
        df['is_top'] = (df['share_rank1'] > 0.5).astype(int)
        log(f"Mean rank: {df[df['avg_rank'] > 0]['avg_rank'].mean():.2f}", f)
        log(f"Share at top rank: {df['is_top'].mean()*100:.1f}%", f)

        # 2.5 Competition clicks
        log("\n--- Building Competition Clicks (C^{(-v)}_st) ---", f)
        total_clicks_st = df.groupby(['session_id', 'year_week'])['C'].sum().reset_index()
        total_clicks_st.columns = ['session_id', 'year_week', 'C_total']
        df = df.merge(total_clicks_st, on=['session_id', 'year_week'], how='left')
        df['C_competition'] = df['C_total'] - df['C']
        df['C_competition'] = df['C_competition'].clip(lower=0)
        log(f"Mean competition clicks: {df['C_competition'].mean():.2f}", f)

        # 2.6 Anchor price
        log("\n--- Building Anchor Price (A_st) ---", f)
        if 'price' in promoted_events.columns:
            anchor_prices = promoted_events.groupby(['user_id', 'year_week']).agg({
                'price': 'mean'
            }).reset_index()
            anchor_prices.columns = ['user_id', 'year_week', 'anchor_price']
            df = df.merge(anchor_prices, on=['user_id', 'year_week'], how='left')
            df['anchor_price'] = df['anchor_price'].fillna(0)
            log(f"Mean anchor price: ${df[df['anchor_price'] > 0]['anchor_price'].mean():.2f}", f)
        else:
            log("Price not available - setting anchor to 0", f)
            df['anchor_price'] = 0

        # 2.7 Interactions
        log("\n--- Building Interaction Terms ---", f)
        df['C_x_short'] = df['C'] * df['is_short']
        df['C_x_top'] = df['C'] * df['is_top']
        log(f"C × Short: mean = {df['C_x_short'].mean():.4f}", f)
        log(f"C × Top: mean = {df['C_x_top'].mean():.4f}", f)

        # ============================================================
        # 3. BASELINE MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 0: BASELINE (User + Week + Vendor FE)", f)
        log("=" * 80, f)

        model0 = pf.feols("Y ~ C | user_id + year_week + vendor_id",
                           data=df, vcov={'CRV1': 'user_id'})
        log(str(model0.summary()), f)
        log(f"\nβ_0 (contemporaneous click effect) = {model0.coef()['C']:.4f}", f)

        # ============================================================
        # 4. AWARENESS/ADSTOCK MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 1: AWARENESS/ADSTOCK (Contemporaneous + Lagged Clicks)", f)
        log("=" * 80, f)
        log("\nH0: β_ℓ = 0 for ℓ ≥ 1 (no delayed/adstock effect)", f)

        lag_vars = ' + '.join([f'C_lag{i}' for i in range(1, MAX_LAG + 1)])
        formula = f"Y ~ C + {lag_vars} | user_id + year_week + vendor_id"
        log(f"\nFormula: {formula}", f)

        model1 = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
        log(str(model1.summary()), f)

        log("\n--- Lag Structure ---", f)
        log(f"β_0 (t=0): {model1.coef()['C']:.4f}", f)
        for lag in range(1, MAX_LAG + 1):
            lag_name = f'C_lag{lag}'
            if lag_name in model1.coef():
                coef = model1.coef()[lag_name]
                se = model1.se()[lag_name]
                log(f"β_{lag} (t-{lag}): {coef:.4f} (SE = {se:.4f}, t = {coef/se:.2f})", f)
            else:
                log(f"β_{lag} (t-{lag}): dropped (no variation)", f)

        # ============================================================
        # 5. VIEW-THROUGH MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 2: VIEW-THROUGH (Clicks + Impressions with Lags)", f)
        log("=" * 80, f)
        log("\nH0: η_ℓ = 0 for all ℓ (no view-through effect)", f)

        model2 = None
        if df['I'].sum() > 0:
            imp_vars = 'I + ' + ' + '.join([f'I_lag{i}' for i in range(1, MAX_LAG + 1)])
            formula = f"Y ~ C + {lag_vars} + {imp_vars} | user_id + year_week + vendor_id"
            log(f"\nFormula: {formula}", f)

            model2 = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
            log(str(model2.summary()), f)

            log("\n--- View-Through Coefficients ---", f)
            if 'I' in model2.coef():
                log(f"η_0 (I, t=0): {model2.coef()['I']:.6f}", f)
            for lag in range(1, MAX_LAG + 1):
                lag_name = f'I_lag{lag}'
                if lag_name in model2.coef():
                    coef = model2.coef()[lag_name]
                    log(f"η_{lag} (I, t-{lag}): {coef:.6f}", f)
        else:
            log("Impressions not available - skipping view-through model", f)

        # ============================================================
        # 6. SEARCH-COST/POSITION MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 3: SEARCH-COST/POSITION (Friction & Position Interactions)", f)
        log("=" * 80, f)
        log("\nH0: δ_1 = δ_2 = 0 (no friction/position moderation)", f)

        formula = f"Y ~ C + C_x_short + C_x_top + {lag_vars} | user_id + year_week + vendor_id"
        log(f"\nFormula: {formula}", f)

        model3 = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
        log(str(model3.summary()), f)

        log("\n--- Interaction Effects ---", f)
        log(f"β (base click effect): {model3.coef()['C']:.4f}", f)
        if 'C_x_short' in model3.coef():
            log(f"δ_1 (C × Short): {model3.coef()['C_x_short']:.4f}", f)
        else:
            log("δ_1 (C × Short): dropped (no variation)", f)
        if 'C_x_top' in model3.coef():
            log(f"δ_2 (C × Top): {model3.coef()['C_x_top']:.4f}", f)
        else:
            log("δ_2 (C × Top): dropped (no variation)", f)

        # ============================================================
        # 7. COMPETITION MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 4: COMPETITION/SUBSTITUTION", f)
        log("=" * 80, f)
        log("\nH0: ρ = 0 (no competition effect)", f)
        log("Expected: ρ < 0 if vendors are substitutes", f)

        formula = f"Y ~ C + C_competition + {lag_vars} | user_id + year_week + vendor_id"
        log(f"\nFormula: {formula}", f)

        model4 = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
        log(str(model4.summary()), f)

        rho = model4.coef()['C_competition']
        se_rho = model4.se()['C_competition']
        log(f"\nρ (competition effect) = {rho:.4f} (SE = {se_rho:.4f})", f)
        if rho < 0:
            log("→ Negative ρ: Clicks on other vendors REDUCE own vendor spend (substitution)", f)
        else:
            log("→ Positive ρ: Clicks on other vendors INCREASE own vendor spend (complementarity)", f)

        # ============================================================
        # 8. ANCHORING MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL 5: ANCHORING", f)
        log("=" * 80, f)
        log("\nH0: κ = 0 (no anchoring effect)", f)

        model5 = None
        if df['anchor_price'].sum() > 0:
            formula = f"Y ~ C + anchor_price + {lag_vars} | user_id + year_week + vendor_id"
            log(f"\nFormula: {formula}", f)

            model5 = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
            log(str(model5.summary()), f)

            kappa = model5.coef()['anchor_price']
            log(f"\nκ (anchoring effect) = {kappa:.6f}", f)
        else:
            log("Anchor price not available - skipping", f)

        # ============================================================
        # 9. FULL UMBRELLA MODEL
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("FULL UMBRELLA MODEL (All Mechanisms)", f)
        log("=" * 80, f)

        terms = ['C']
        terms.extend([f'C_lag{i}' for i in range(1, MAX_LAG + 1)])
        if df['I'].sum() > 0:
            terms.append('I')
            terms.extend([f'I_lag{i}' for i in range(1, MAX_LAG + 1)])
        terms.extend(['C_x_short', 'C_x_top'])
        terms.append('C_competition')
        if df['anchor_price'].sum() > 0:
            terms.append('anchor_price')

        formula = f"Y ~ {' + '.join(terms)} | user_id + year_week + vendor_id"
        log(f"\nFormula: Y ~ {' + '.join(terms[:5])} + ...", f)
        log(f"         ... + {' + '.join(terms[5:])}", f)
        log(f"         | user_id + year_week + vendor_id", f)

        model_full = pf.feols(formula, data=df, vcov={'CRV1': 'user_id'})
        log(str(model_full.summary()), f)

        # Coefficient tests
        log("\n" + "=" * 80, f)
        log("COEFFICIENT TESTS (Full Umbrella Model)", f)
        log("=" * 80, f)

        coefs = model_full.coef()
        ses = model_full.se()

        def test_coef(name, expected_sign=None):
            if name in coefs:
                c, s = coefs[name], ses[name]
                if s > 0:
                    t = c / s
                    sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.65 else ''
                    sign_match = ''
                    if expected_sign == '+' and c > 0:
                        sign_match = ' [expected]'
                    elif expected_sign == '-' and c < 0:
                        sign_match = ' [expected]'
                    elif expected_sign:
                        sign_match = ' [unexpected]'
                    log(f"{name:20s}: {c:10.4f} (t={t:6.2f}){sig}{sign_match}", f)
                else:
                    log(f"{name:20s}: {c:10.4f} (SE=0, dropped)", f)
            else:
                log(f"{name:20s}: not in model", f)

        log("\n--- Short-Run Activation ---", f)
        test_coef('C', '+')

        log("\n--- Awareness/Adstock (Lagged Effects) ---", f)
        for lag in range(1, MAX_LAG + 1):
            test_coef(f'C_lag{lag}')

        log("\n--- View-Through (Impressions) ---", f)
        test_coef('I')
        for lag in range(1, MAX_LAG + 1):
            test_coef(f'I_lag{lag}')

        log("\n--- Search-Cost/Position ---", f)
        test_coef('C_x_short')
        test_coef('C_x_top')

        log("\n--- Competition/Substitution ---", f)
        test_coef('C_competition', '-')

        log("\n--- Anchoring ---", f)
        test_coef('anchor_price')

        # ============================================================
        # 10. MODEL COMPARISON
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("MODEL COMPARISON", f)
        log("=" * 80, f)

        models = {
            'M0: Baseline': model0,
            'M1: + Lags': model1,
            'M3: + Interactions': model3,
            'M4: + Competition': model4,
            'Full': model_full
        }
        if model2 is not None:
            models['M2: + Impressions'] = model2
        if model5 is not None:
            models['M5: + Anchoring'] = model5

        log(f"\n{'Model':<25s} {'β_C':<10s} {'R²':<10s} {'N':<10s}", f)
        log("-" * 55, f)

        for name, m in models.items():
            beta_c = m.coef()['C']
            r2 = m.r2 if hasattr(m, 'r2') else 'N/A'
            n = m._N if hasattr(m, '_N') else 'N/A'
            if isinstance(r2, float) and isinstance(n, int):
                log(f"{name:<25s} {beta_c:<10.4f} {r2:<10.4f} {n:<10,}", f)
            else:
                log(f"{name:<25s} {beta_c:<10.4f} {str(r2):<10s} {str(n):<10s}", f)

        # ============================================================
        # 11. SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("UMBRELLA REGRESSION SUMMARY", f)
        log("=" * 80, f)

        log("\n--- Key Findings ---", f)

        beta0 = model_full.coef()['C']
        se0 = model_full.se()['C']
        log(f"\n1. SHORT-RUN ACTIVATION", f)
        log(f"   β_0 = {beta0:.4f} (t = {beta0/se0:.2f})", f)
        if abs(beta0/se0) > 1.96:
            log(f"   → REJECT H0: Significant contemporaneous click effect", f)

        log(f"\n2. AWARENESS/ADSTOCK", f)
        any_lag_sig = False
        for lag in range(1, MAX_LAG + 1):
            name = f'C_lag{lag}'
            if name in model_full.coef():
                c, s = model_full.coef()[name], model_full.se()[name]
                if s > 0 and abs(c/s) > 1.96:
                    any_lag_sig = True
                    log(f"   β_{lag} = {c:.4f} (t = {c/s:.2f}) - SIGNIFICANT", f)
        if not any_lag_sig:
            log("   → No significant lagged effects (limited adstock)", f)

        if 'C_competition' in model_full.coef():
            rho = model_full.coef()['C_competition']
            se_rho = model_full.se()['C_competition']
            log(f"\n3. COMPETITION/SUBSTITUTION", f)
            log(f"   ρ = {rho:.4f} (t = {rho/se_rho:.2f})", f)
            if rho < 0 and abs(rho/se_rho) > 1.96:
                log("   → SUBSTITUTION: Clicks on other vendors reduce own vendor spend", f)
            elif rho > 0 and abs(rho/se_rho) > 1.96:
                log("   → COMPLEMENTARITY: Clicks on other vendors increase own vendor spend", f)

        log("\n--- Mechanism Summary ---", f)
        log("Supported mechanisms (|t| > 1.96):", f)
        for name in model_full.coef().keys():
            c, s = model_full.coef()[name], model_full.se()[name]
            if s > 0 and abs(c/s) > 1.96:
                log(f"  ✓ {name}: {c:.4f}", f)

        # Save results
        umbrella_results = {
            'gap_days': GAP_DAYS,
            'n_obs': len(df),
            'n_sessions': df['session_id'].nunique(),
            'n_users': df['user_id'].nunique(),
            'baseline_beta': model0.coef()['C'],
            'full_model_coefs': {k: float(v) for k, v in model_full.coef().items()},
            'full_model_ses': {k: float(v) for k, v in model_full.se().items()}
        }

        with open(DATA_DIR / 'umbrella_results.json', 'w') as jf:
            json.dump(umbrella_results, jf, indent=2)

        log(f"\nResults saved to {DATA_DIR / 'umbrella_results.json'}", f)

        log("\n" + "=" * 80, f)
        log("UMBRELLA REGRESSION COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
