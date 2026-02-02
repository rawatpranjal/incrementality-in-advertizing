#!/usr/bin/env python3
"""
02_position_effects_analysis.py
Comprehensive EDA + Position effects models.
All tables printed to stdout, no opinions.

References:
- Wang et al. (2018) - Position Bias EM/IPW (WSDM)
- Ai et al. (2018) - Dual Learning Algorithm (SIGIR)
- Dupret & Piwowarski (2008) - User Browsing Model (SIGIR)
- Narayanan & Kalyanam (2015) - Regression Discontinuity (Marketing Science)
- Jenkins (1995) - Discrete-Time Survival Models (Oxford Bull Econ Stat)
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "02_position_effects_analysis.txt"


def log(msg, f):
    print(msg)
    f.write(msg + "\n")


def main():
    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("POSITION EFFECTS: COMPREHENSIVE EDA + MODELS", f)
        log("=" * 80, f)
        log(f"Timestamp: {datetime.now()}", f)

        # =================================================================
        # LOAD DATA
        # =================================================================
        log("\n" + "=" * 80, f)
        log("DATA LOADING", f)
        log("=" * 80, f)

        ar = pd.read_parquet(DATA_DIR / "auctions_results_p5.parquet")
        au = pd.read_parquet(DATA_DIR / "auctions_users_p5.parquet")
        imps = pd.read_parquet(DATA_DIR / "impressions_p5.parquet")
        clicks = pd.read_parquet(DATA_DIR / "clicks_p5.parquet")
        catalog = pd.read_parquet(DATA_DIR / "catalog_p5.parquet")

        log(f"auctions_users: {len(au):,} rows, {au['AUCTION_ID'].nunique():,} auctions, {au['USER_ID'].nunique():,} users", f)
        log(f"auctions_results: {len(ar):,} rows", f)
        log(f"impressions: {len(imps):,} rows", f)
        log(f"clicks: {len(clicks):,} rows", f)
        log(f"catalog: {len(catalog):,} rows", f)

        # Build panel
        log("\n--- Building Panel ---", f)
        df = ar.merge(au[['AUCTION_ID', 'USER_ID']], on='AUCTION_ID', how='left')

        imp_keys = set(zip(imps['AUCTION_ID'], imps['PRODUCT_ID']))
        click_keys = set(zip(clicks['AUCTION_ID'], clicks['PRODUCT_ID']))

        df['GOT_IMPRESSION'] = df.apply(lambda r: (r['AUCTION_ID'], r['PRODUCT_ID']) in imp_keys, axis=1).astype(int)
        df['GOT_CLICK'] = df.apply(lambda r: (r['AUCTION_ID'], r['PRODUCT_ID']) in click_keys, axis=1).astype(int)

        log(f"Panel: {len(df):,} bid-level rows", f)
        log(f"Unique auctions: {df['AUCTION_ID'].nunique():,}", f)
        log(f"Unique users: {df['USER_ID'].nunique():,}", f)
        log(f"Unique products: {df['PRODUCT_ID'].nunique():,}", f)
        log(f"Unique vendors: {df['VENDOR_ID'].nunique():,}", f)
        log(f"Unique campaigns: {df['CAMPAIGN_ID'].nunique():,}", f)

        # =================================================================
        # EDA 1: BASIC FUNNEL AND POSITION DISTRIBUTION
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 1: BASIC FUNNEL AND POSITION DISTRIBUTION", f)
        log("=" * 80, f)

        log("\n--- Overall Funnel ---", f)
        log(f"Total bids: {len(df):,}", f)
        log(f"Bids with IS_WINNER=True: {df['IS_WINNER'].sum():,} ({df['IS_WINNER'].mean():.1%})", f)
        log(f"Bids with impression: {df['GOT_IMPRESSION'].sum():,} ({df['GOT_IMPRESSION'].mean():.1%})", f)
        log(f"Bids with click: {df['GOT_CLICK'].sum():,} ({df['GOT_CLICK'].mean():.2%})", f)
        if df['GOT_IMPRESSION'].sum() > 0:
            log(f"CTR given impression: {df[df['GOT_IMPRESSION']==1]['GOT_CLICK'].mean():.2%}", f)

        log("\n--- Click Rate by RANKING (1-30) ---", f)
        rank_ctr = df.groupby('RANKING').agg({
            'GOT_CLICK': ['sum', 'mean', 'count'],
            'GOT_IMPRESSION': ['sum', 'mean'],
            'IS_WINNER': 'mean'
        })
        rank_ctr.columns = ['clicks', 'ctr', 'n_bids', 'impressions', 'imp_rate', 'win_rate']
        rank_ctr['ctr_pct'] = (rank_ctr['ctr'] * 100).round(4)
        rank_ctr['imp_rate_pct'] = (rank_ctr['imp_rate'] * 100).round(2)
        log(rank_ctr.head(30).to_string(), f)

        log("\n--- Slots per Auction (IS_WINNER=True count) ---", f)
        slots_per_auction = df.groupby('AUCTION_ID')['IS_WINNER'].sum()
        log(f"Mean: {slots_per_auction.mean():.1f}", f)
        log(f"Median: {slots_per_auction.median():.0f}", f)
        log(f"Min: {slots_per_auction.min():.0f}", f)
        log(f"Max: {slots_per_auction.max():.0f}", f)
        log(f"p10: {slots_per_auction.quantile(0.1):.0f}", f)
        log(f"p90: {slots_per_auction.quantile(0.9):.0f}", f)

        log("\n--- Max Rank with Impression ---", f)
        max_imp_rank = df[df['GOT_IMPRESSION'] == 1]['RANKING'].max()
        log(f"Max rank that got impression: {max_imp_rank}", f)
        imp_rank_dist = df[df['GOT_IMPRESSION'] == 1]['RANKING'].describe()
        log(f"Impression rank distribution:\n{imp_rank_dist.to_string()}", f)

        log("\n--- Bids per Auction ---", f)
        bids_per_auction = df.groupby('AUCTION_ID').size()
        log(f"Mean: {bids_per_auction.mean():.1f}", f)
        log(f"Median: {bids_per_auction.median():.0f}", f)
        log(f"Min: {bids_per_auction.min():.0f}", f)
        log(f"Max: {bids_per_auction.max():.0f}", f)

        # =================================================================
        # EDA 2: SAME PRODUCT AT DIFFERENT RANKS
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 2: IDENTIFICATION CHECK — SAME PRODUCT AT DIFFERENT RANKS", f)
        log("=" * 80, f)

        log("\n--- Product Rank Variation ---", f)
        product_ranks = df.groupby('PRODUCT_ID')['RANKING'].agg(['nunique', 'min', 'max', 'mean', 'count'])
        product_ranks.columns = ['n_distinct_ranks', 'min_rank', 'max_rank', 'mean_rank', 'n_bids']
        log(f"Products appearing at 1 rank only: {(product_ranks['n_distinct_ranks'] == 1).sum():,}", f)
        log(f"Products appearing at 2+ ranks: {(product_ranks['n_distinct_ranks'] >= 2).sum():,}", f)
        log(f"Products appearing at 5+ ranks: {(product_ranks['n_distinct_ranks'] >= 5).sum():,}", f)
        log(f"Products appearing at 10+ ranks: {(product_ranks['n_distinct_ranks'] >= 10).sum():,}", f)

        log("\n--- Product-Level CTR vs Average Rank ---", f)
        product_ctr = df.groupby('PRODUCT_ID').agg({
            'GOT_CLICK': ['sum', 'mean'],
            'RANKING': 'mean',
            'AUCTION_ID': 'count'
        })
        product_ctr.columns = ['clicks', 'ctr', 'avg_rank', 'n_bids']
        product_ctr = product_ctr[product_ctr['n_bids'] >= 5]  # filter for stability
        if len(product_ctr) > 0:
            corr = product_ctr['ctr'].corr(product_ctr['avg_rank'])
            log(f"Correlation(CTR, avg_rank): {corr:.4f} (N={len(product_ctr):,} products with 5+ bids)", f)

        log("\n--- Top 20 Products by Bid Count ---", f)
        log(product_ctr.nlargest(20, 'n_bids').to_string(), f)

        # =================================================================
        # EDA 3: AUCTION SCORE CONSTRUCTION FOR RDD
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 3: AUCTION SCORE CONSTRUCTION FOR RDD", f)
        log("=" * 80, f)

        log("\n--- Score = QUALITY × FINAL_BID ---", f)
        df['SCORE'] = df['QUALITY'] * df['FINAL_BID']
        log(f"Score distribution:", f)
        log(df['SCORE'].describe().to_string(), f)

        log("\n--- Rank vs Score Correlation ---", f)
        # Within each auction, check if score ordering matches ranking
        def check_score_rank_match(grp):
            grp_sorted_by_score = grp.sort_values('SCORE', ascending=False).reset_index(drop=True)
            grp_sorted_by_rank = grp.sort_values('RANKING').reset_index(drop=True)
            match = (grp_sorted_by_score['PRODUCT_ID'].values == grp_sorted_by_rank['PRODUCT_ID'].values).mean()
            return match

        rank_score_match = df.groupby('AUCTION_ID').apply(check_score_rank_match)
        log(f"Fraction of auctions where score ordering matches rank: {rank_score_match.mean():.2%}", f)

        log("\n--- Score Margin to Next Competitor ---", f)
        def compute_margin(grp):
            grp = grp.sort_values('RANKING')
            grp['score_margin'] = grp['SCORE'].diff(-1)  # margin to next rank
            return grp

        df_margin = df.groupby('AUCTION_ID', group_keys=False).apply(compute_margin)
        df_margin = df_margin[df_margin['score_margin'].notna()]
        log(f"Score margin distribution:\n{df_margin['score_margin'].describe().to_string()}", f)

        log("\n--- Score Margin vs Click Outcome ---", f)
        margin_bins = pd.qcut(df_margin['score_margin'], q=5, duplicates='drop')
        margin_ctr = df_margin.groupby(margin_bins, observed=True)['GOT_CLICK'].mean()
        log(f"CTR by score margin quintile:\n{margin_ctr.to_string()}", f)

        log("\n--- Bunching Check (FINAL_BID round numbers) ---", f)
        df['bid_mod_100'] = df['FINAL_BID'] % 100
        df['bid_mod_1000'] = df['FINAL_BID'] % 1000
        log(f"Bids ending in 00: {(df['bid_mod_100'] == 0).sum():,} ({(df['bid_mod_100'] == 0).mean():.1%})", f)
        log(f"Bids ending in 000: {(df['bid_mod_1000'] == 0).sum():,} ({(df['bid_mod_1000'] == 0).mean():.1%})", f)

        # =================================================================
        # EDA 4: SURVIVAL/CASCADE STRUCTURE
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 4: SURVIVAL/CASCADE STRUCTURE", f)
        log("=" * 80, f)

        log("\n--- Click Depth: Lowest Rank with Click per Auction ---", f)
        auction_click_depth = df[df['GOT_CLICK'] == 1].groupby('AUCTION_ID')['RANKING'].min()
        if len(auction_click_depth) > 0:
            log(f"Click depth distribution:\n{auction_click_depth.describe().to_string()}", f)
            log(f"\nClick depth frequency:", f)
            click_depth_freq = auction_click_depth.value_counts().sort_index().head(20)
            log(click_depth_freq.to_string(), f)
        else:
            log("No clicks in data", f)

        log("\n--- Auctions with Clicks vs No Clicks ---", f)
        auctions_with_click = df[df['GOT_CLICK'] == 1]['AUCTION_ID'].nunique()
        auctions_total = df['AUCTION_ID'].nunique()
        log(f"Auctions with at least 1 click: {auctions_with_click:,} ({auctions_with_click/auctions_total:.1%})", f)
        log(f"Auctions with no clicks: {auctions_total - auctions_with_click:,}", f)

        log("\n--- Cascade Test: Users who click rank 1 — do they see rank 5+? ---", f)
        users_click_rank1 = df[(df['GOT_CLICK'] == 1) & (df['RANKING'] == 1)]['USER_ID'].unique()
        if len(users_click_rank1) > 0:
            imps_users_click1 = df[(df['USER_ID'].isin(users_click_rank1)) & (df['GOT_IMPRESSION'] == 1)]
            rank5plus = imps_users_click1[imps_users_click1['RANKING'] >= 5]
            log(f"Users who clicked at rank 1: {len(users_click_rank1):,}", f)
            log(f"Of those, impressions at rank 5+: {len(rank5plus):,}", f)

        # =================================================================
        # EDA 5: USER HETEROGENEITY
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 5: USER HETEROGENEITY", f)
        log("=" * 80, f)

        log("\n--- Clicks per User Distribution ---", f)
        user_clicks = df.groupby('USER_ID')['GOT_CLICK'].sum()
        log(f"Clicks per user distribution:\n{user_clicks.describe().to_string()}", f)
        log(f"\nUsers with 0 clicks: {(user_clicks == 0).sum():,}", f)
        log(f"Users with 1 click: {(user_clicks == 1).sum():,}", f)
        log(f"Users with 2+ clicks: {(user_clicks >= 2).sum():,}", f)
        log(f"Users with 5+ clicks: {(user_clicks >= 5).sum():,}", f)

        log("\n--- Rank Distribution: Users with 0 vs 1+ Clicks ---", f)
        users_0_clicks = user_clicks[user_clicks == 0].index
        users_1plus_clicks = user_clicks[user_clicks >= 1].index

        if len(users_0_clicks) > 0:
            avg_rank_0clicks = df[df['USER_ID'].isin(users_0_clicks)]['RANKING'].mean()
            log(f"Avg rank seen by users with 0 clicks: {avg_rank_0clicks:.2f}", f)
        if len(users_1plus_clicks) > 0:
            avg_rank_1plus = df[df['USER_ID'].isin(users_1plus_clicks)]['RANKING'].mean()
            log(f"Avg rank seen by users with 1+ clicks: {avg_rank_1plus:.2f}", f)

        log("\n--- Bids per User ---", f)
        user_bids = df.groupby('USER_ID').size()
        log(f"Bids per user distribution:\n{user_bids.describe().to_string()}", f)

        log("\n--- Impressions per User ---", f)
        user_imps = df.groupby('USER_ID')['GOT_IMPRESSION'].sum()
        log(f"Impressions per user distribution:\n{user_imps.describe().to_string()}", f)

        # =================================================================
        # EDA 6: PRODUCT/AUCTION CHARACTERISTICS
        # =================================================================
        log("\n" + "=" * 80, f)
        log("EDA 6: PRODUCT/AUCTION CHARACTERISTICS", f)
        log("=" * 80, f)

        log("\n--- QUALITY Distribution ---", f)
        log(df['QUALITY'].describe().to_string(), f)
        log(f"\nQUALITY percentiles:", f)
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            log(f"  p{p}: {df['QUALITY'].quantile(p/100):.6f}", f)

        log("\n--- FINAL_BID Distribution (in cents) ---", f)
        log(df['FINAL_BID'].describe().to_string(), f)
        log(f"\nFINAL_BID in dollars:", f)
        log((df['FINAL_BID'] / 100).describe().to_string(), f)

        log("\n--- CONVERSION_RATE Distribution ---", f)
        log(df['CONVERSION_RATE'].describe().to_string(), f)

        log("\n--- PACING Distribution ---", f)
        log(df['PACING'].describe().to_string(), f)

        log("\n--- Correlations: QUALITY vs RANKING vs CTR ---", f)
        corr_quality_rank = df['QUALITY'].corr(df['RANKING'])
        corr_quality_ctr = df['QUALITY'].corr(df['GOT_CLICK'])
        corr_bid_rank = df['FINAL_BID'].corr(df['RANKING'])
        corr_bid_ctr = df['FINAL_BID'].corr(df['GOT_CLICK'])
        log(f"Corr(QUALITY, RANKING): {corr_quality_rank:.4f}", f)
        log(f"Corr(QUALITY, GOT_CLICK): {corr_quality_ctr:.4f}", f)
        log(f"Corr(FINAL_BID, RANKING): {corr_bid_rank:.4f}", f)
        log(f"Corr(FINAL_BID, GOT_CLICK): {corr_bid_ctr:.4f}", f)

        log("\n--- PRICE Effect on Clicks by Rank ---", f)
        df['price_quintile'] = pd.qcut(df['PRICE'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        price_rank_ctr = df.groupby(['price_quintile', 'RANKING'], observed=True)['GOT_CLICK'].mean().unstack(level=0)
        log(f"CTR by price quintile and rank (top 10 ranks):\n{price_rank_ctr.head(10).to_string()}", f)

        log("\n--- CONVERSION_RATE vs Observed Click Rate ---", f)
        df['cvr_quintile'] = pd.qcut(df['CONVERSION_RATE'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        cvr_ctr = df.groupby('cvr_quintile', observed=True)['GOT_CLICK'].mean()
        log(f"CTR by CONVERSION_RATE quintile:\n{cvr_ctr.to_string()}", f)

        # =================================================================
        # MODEL 0: REDUCED FORM
        # =================================================================
        log("\n" + "=" * 80, f)
        log("MODEL 0: REDUCED FORM (RAW POSITION EFFECTS)", f)
        log("=" * 80, f)

        log("\n--- CTR by Rank (Full Table) ---", f)
        rank_stats = df.groupby('RANKING').agg({
            'GOT_IMPRESSION': ['sum', 'mean'],
            'GOT_CLICK': ['sum', 'mean'],
            'IS_WINNER': 'sum',
            'AUCTION_ID': 'count'
        })
        rank_stats.columns = ['impressions', 'imp_rate', 'clicks', 'ctr', 'winners', 'n_bids']
        rank_stats['ctr_given_imp'] = rank_stats['clicks'] / rank_stats['impressions'].replace(0, np.nan)
        log(rank_stats.head(50).to_string(), f)

        # =================================================================
        # MODEL 1: CASCADE MODEL (Dupret & Piwowarski 2008)
        # =================================================================
        log("\n" + "=" * 80, f)
        log("MODEL 1: CASCADE / USER BROWSING MODEL (Dupret & Piwowarski 2008)", f)
        log("=" * 80, f)

        auction_clicks_df = df[df['GOT_CLICK'] == 1].groupby('AUCTION_ID')['RANKING'].min().reset_index()
        auction_clicks_df.columns = ['AUCTION_ID', 'first_click_rank']

        df_cascade = df.merge(auction_clicks_df, on='AUCTION_ID', how='left')
        df_cascade['first_click_rank'] = df_cascade['first_click_rank'].fillna(999)
        df_cascade['EXAMINED'] = (df_cascade['RANKING'] <= df_cascade['first_click_rank']).astype(int)

        log(f"\nTotal bids: {len(df_cascade):,}", f)
        log(f"Examined (rank <= first click): {df_cascade['EXAMINED'].sum():,} ({df_cascade['EXAMINED'].mean():.1%})", f)

        exam_by_rank = df_cascade.groupby('RANKING').agg({
            'EXAMINED': 'mean',
            'GOT_CLICK': 'sum',
            'AUCTION_ID': 'count'
        }).rename(columns={'AUCTION_ID': 'n'})

        examined_bids = df_cascade[df_cascade['EXAMINED'] == 1]
        click_given_exam = examined_bids.groupby('RANKING').agg({'GOT_CLICK': ['sum', 'count']})
        click_given_exam.columns = ['clicks', 'examined']
        click_given_exam['p_click_given_exam'] = click_given_exam['clicks'] / click_given_exam['examined']

        log("\n--- Cascade Model Estimates ---", f)
        log("Rank | P(Examine) | P(Click|Exam) | Raw CTR | N", f)
        log("-" * 60, f)
        for r in range(1, 31):
            if r in exam_by_rank.index and r in click_given_exam.index:
                p_exam = exam_by_rank.loc[r, 'EXAMINED']
                p_click_exam = click_given_exam.loc[r, 'p_click_given_exam']
                raw_ctr = exam_by_rank.loc[r, 'GOT_CLICK'] / exam_by_rank.loc[r, 'n']
                n = exam_by_rank.loc[r, 'n']
                log(f"{r:4d} | {p_exam:10.2%} | {p_click_exam:13.4%} | {raw_ctr:7.4%} | {n:,}", f)

        # =================================================================
        # MODEL 2: IPW (Wang et al. 2018)
        # =================================================================
        log("\n" + "=" * 80, f)
        log("MODEL 2: INVERSE PROPENSITY WEIGHTING (Wang et al. 2018)", f)
        log("=" * 80, f)

        propensity = exam_by_rank['EXAMINED'].to_dict()
        max_rank_data = max(propensity.keys())
        for r in range(1, 100):
            if r not in propensity:
                propensity[r] = propensity.get(max_rank_data, 0.01)

        df_ipw = df.copy()
        df_ipw['propensity'] = df_ipw['RANKING'].map(propensity).clip(lower=0.01)
        df_ipw['ipw_weight'] = 1 / df_ipw['propensity']
        df_ipw['weighted_click'] = df_ipw['GOT_CLICK'] * df_ipw['ipw_weight']

        ipw_by_rank = df_ipw.groupby('RANKING').agg({
            'GOT_CLICK': 'sum',
            'weighted_click': 'sum',
            'AUCTION_ID': 'count'
        }).rename(columns={'AUCTION_ID': 'n'})
        ipw_by_rank['raw_ctr'] = ipw_by_rank['GOT_CLICK'] / ipw_by_rank['n']
        ipw_by_rank['ipw_ctr'] = ipw_by_rank['weighted_click'] / ipw_by_rank['n']

        log("\n--- IPW-Adjusted CTR ---", f)
        log("Rank | Propensity | Raw CTR | IPW CTR | N", f)
        log("-" * 55, f)
        for r in range(1, 31):
            if r in ipw_by_rank.index:
                row = ipw_by_rank.loc[r]
                prop = propensity.get(r, np.nan)
                log(f"{r:4d} | {prop:10.2%} | {row['raw_ctr']:7.4%} | {row['ipw_ctr']:7.4%} | {int(row['n']):,}", f)

        log(f"\nOverall Raw CTR: {df_ipw['GOT_CLICK'].mean():.4%}", f)
        log(f"Overall IPW CTR: {df_ipw['weighted_click'].sum() / len(df_ipw):.4%}", f)

        # =================================================================
        # MODEL 3: DISCRETE-TIME SURVIVAL (Jenkins 1995)
        # =================================================================
        log("\n" + "=" * 80, f)
        log("MODEL 3: DISCRETE-TIME SURVIVAL MODEL (Jenkins 1995)", f)
        log("=" * 80, f)

        max_rank_surv = 30
        survival_rows = []
        for auction_id, grp in tqdm(df.groupby('AUCTION_ID'), desc="Building survival data"):
            grp_sorted = grp.sort_values('RANKING')
            for _, row in grp_sorted.iterrows():
                if row['RANKING'] > max_rank_surv:
                    break
                survival_rows.append({
                    'AUCTION_ID': auction_id,
                    'RANKING': row['RANKING'],
                    'CLICK': row['GOT_CLICK'],
                    'QUALITY': row['QUALITY'],
                    'FINAL_BID': row['FINAL_BID'],
                    'CONVERSION_RATE': row['CONVERSION_RATE']
                })
                if row['GOT_CLICK'] == 1:
                    break

        df_surv = pd.DataFrame(survival_rows)
        log(f"\nSurvival dataset: {len(df_surv):,} person-rank observations", f)

        df_surv['log_rank'] = np.log(df_surv['RANKING'])
        df_surv['log_bid'] = np.log(df_surv['FINAL_BID'].clip(lower=1))
        df_surv['log_quality'] = np.log(df_surv['QUALITY'].clip(lower=0.001))

        try:
            model_rank = smf.logit('CLICK ~ log_rank', data=df_surv).fit(disp=0)
            log("\n--- Hazard Model 1: Rank Only ---", f)
            log(str(model_rank.summary().tables[1]), f)
        except Exception as e:
            log(f"Model 1 failed: {e}", f)

        try:
            model_full = smf.logit('CLICK ~ log_rank + log_quality + log_bid', data=df_surv).fit(disp=0)
            log("\n--- Hazard Model 2: Rank + Quality + Bid ---", f)
            log(str(model_full.summary().tables[1]), f)
        except Exception as e:
            log(f"Model 2 failed: {e}", f)

        hazard_by_rank = df_surv.groupby('RANKING').agg({'CLICK': ['sum', 'count']})
        hazard_by_rank.columns = ['clicks', 'at_risk']
        hazard_by_rank['hazard'] = hazard_by_rank['clicks'] / hazard_by_rank['at_risk']
        hazard_by_rank['survival'] = (1 - hazard_by_rank['hazard']).cumprod()

        log("\n--- Hazard and Survival by Rank ---", f)
        log("Rank | Hazard h(k) | Survival S(k) | At Risk | Clicks", f)
        log("-" * 60, f)
        for r in range(1, min(31, max_rank_surv + 1)):
            if r in hazard_by_rank.index:
                row = hazard_by_rank.loc[r]
                log(f"{r:4d} | {row['hazard']:11.4%} | {row['survival']:13.4%} | {int(row['at_risk']):7,} | {int(row['clicks']):6,}", f)

        # =================================================================
        # MODEL 4: REGRESSION DISCONTINUITY (Narayanan & Kalyanam 2015)
        # =================================================================
        log("\n" + "=" * 80, f)
        log("MODEL 4: REGRESSION DISCONTINUITY (Narayanan & Kalyanam 2015)", f)
        log("=" * 80, f)

        log("\n--- RD Estimates at Multiple Boundaries ---", f)
        log("Boundary | CTR(lower) | CTR(higher) | Effect | N_lower | N_higher", f)
        log("-" * 70, f)
        for cutoff in range(1, 21):
            lower = df[df['RANKING'] == cutoff]
            higher = df[df['RANKING'] == cutoff + 1]
            if len(lower) > 0 and len(higher) > 0:
                ctr_lower = lower['GOT_CLICK'].mean()
                ctr_higher = higher['GOT_CLICK'].mean()
                log(f"{cutoff:2d} vs {cutoff+1:2d} | {ctr_lower:10.4%} | {ctr_higher:11.4%} | {ctr_lower - ctr_higher:+7.4%} | {len(lower):7,} | {len(higher):8,}", f)

        # Close-race RD at rank 1/2
        auctions_12 = df[df['RANKING'].isin([1, 2])].groupby('AUCTION_ID').filter(lambda x: len(x) == 2)
        if len(auctions_12) > 0:
            rd_pivot = auctions_12.pivot_table(
                index='AUCTION_ID', columns='RANKING',
                values=['FINAL_BID', 'GOT_CLICK', 'QUALITY', 'SCORE'], aggfunc='first'
            )
            rd_pivot.columns = ['_'.join(map(str, c)) for c in rd_pivot.columns]

            if 'FINAL_BID_1' in rd_pivot.columns and 'FINAL_BID_2' in rd_pivot.columns:
                rd_pivot['bid_gap'] = rd_pivot['FINAL_BID_1'] - rd_pivot['FINAL_BID_2']

                log(f"\n--- RD at Rank 1/2 Boundary ---", f)
                log(f"Auctions with both rank 1 and 2: {len(rd_pivot):,}", f)
                log(f"Bid gap (rank1 - rank2): mean=${rd_pivot['bid_gap'].mean()/100:.4f}, median=${rd_pivot['bid_gap'].median()/100:.4f}", f)

                ctr1 = rd_pivot['GOT_CLICK_1'].mean()
                ctr2 = rd_pivot['GOT_CLICK_2'].mean()
                log(f"CTR at Rank 1: {ctr1:.4%}", f)
                log(f"CTR at Rank 2: {ctr2:.4%}", f)
                log(f"RD Effect: {ctr1 - ctr2:.4%}", f)

                # Close races
                close = rd_pivot[rd_pivot['bid_gap'].abs() < rd_pivot['bid_gap'].abs().quantile(0.25)]
                if len(close) > 10:
                    log(f"\n--- Close Races (bottom 25% bid gap, N={len(close):,}) ---", f)
                    log(f"CTR at Rank 1: {close['GOT_CLICK_1'].mean():.4%}", f)
                    log(f"CTR at Rank 2: {close['GOT_CLICK_2'].mean():.4%}", f)
                    log(f"RD Effect: {close['GOT_CLICK_1'].mean() - close['GOT_CLICK_2'].mean():.4%}", f)

        # =================================================================
        # REFERENCES
        # =================================================================
        log("\n" + "=" * 80, f)
        log("REFERENCES", f)
        log("=" * 80, f)
        log("Wang et al. (2018): https://dl.acm.org/doi/10.1145/3159652.3159732", f)
        log("Ai et al. (2018): https://arxiv.org/abs/1804.05938", f)
        log("Dupret & Piwowarski (2008): https://www.piwowarski.fr/publication/dupret_user_2008-qkt68tna/dupret_user_2008-QKT68TNA.pdf", f)
        log("Narayanan & Kalyanam (2015): https://pubsonline.informs.org/doi/10.1287/mksc.2014.0893", f)
        log("Jenkins (1995): https://www.stata.com/manuals13/stdiscrete.pdf", f)

        log("\n" + "=" * 80, f)
        log("DONE", f)
        log("=" * 80, f)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
