#!/usr/bin/env python3
"""
11_semantic_halo.py
Tests if ads "inspire" purchases of similar products (even from different vendors).
Uses text similarity on product names to match shown ads to purchased items.
Compares H_shown (max similarity to displayed ads) vs H_not_shown (max similarity
to non-displayed bids in same auctions) as a control for intent.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("/Users/pranjal/Code/marketplace-incrementality/eda/data")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "11_semantic_halo.txt"

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def main():
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("11_SEMANTIC_HALO", f)
        log("=" * 80, f)
        log("", f)
        log("RESEARCH HYPOTHESES:", f)
        log("", f)
        log("This script tests whether ads function as 'visual search' - inspiring purchases", f)
        log("of similar products even when the user doesn't click the ad or buys from a", f)
        log("different vendor. The key insight is that a user sees a 'Vintage Leather Jacket'", f)
        log("for $100, likes the style, then searches and buys a similar one for $60 from", f)
        log("a competitor. Current attribution counts this as 'failed ad' when the ad", f)
        log("actually sparked the intent.", f)
        log("", f)
        log("We test this by computing text similarity between purchased items and shown", f)
        log("ads. As a control, we also compute similarity to NOT-shown bids (rank > K)", f)
        log("from the same auctions. If H_shown > H_not_shown systematically, ads are", f)
        log("driving purchase decisions beyond mere intent matching.", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # ============================================================
        # 1. LOAD DATA
        # ============================================================
        log("LOADING DATA", f)
        log("-" * 40, f)

        log("\nLoading catalog...", f)
        catalog = pd.read_parquet(SOURCE_DIR / 'catalog_365d.parquet')
        log(f"  Catalog rows: {len(catalog):,}", f)
        log(f"  Columns: {list(catalog.columns)}", f)

        log("\nLoading auction results (bids)...", f)
        bids = pd.read_parquet(SOURCE_DIR / 'auctions_results_365d.parquet')
        log(f"  Bids: {len(bids):,}", f)

        log("\nLoading auctions_users...", f)
        auctions_users = pd.read_parquet(SOURCE_DIR / 'auctions_users_365d.parquet')
        log(f"  Auctions: {len(auctions_users):,}", f)

        log("\nLoading purchases_mapped...", f)
        purchases = pd.read_parquet(DATA_DIR / 'purchases_mapped.parquet')
        valid_purchases = purchases[purchases['is_post_click']].copy()
        log(f"  Valid purchases: {len(valid_purchases):,}", f)

        log("\nLoading events_with_sessions...", f)
        events = pd.read_parquet(DATA_DIR / 'events_with_sessions.parquet')
        log(f"  Events: {len(events):,}", f)

        log("\nLoading auction_panel (for K thresholds)...", f)
        auction_panel = pd.read_parquet(DATA_DIR / 'auction_panel.parquet')
        log(f"  Auction-vendor pairs: {len(auction_panel):,}", f)

        # ============================================================
        # 2. BUILD TEXT SIMILARITY INFRASTRUCTURE
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("BUILDING TEXT SIMILARITY INFRASTRUCTURE", f)
        log("=" * 80, f)

        # Get product names from catalog
        catalog_names = catalog[['PRODUCT_ID', 'NAME']].dropna(subset=['NAME']).copy()
        catalog_names['NAME'] = catalog_names['NAME'].astype(str)
        log(f"\nProducts with NAME: {len(catalog_names):,}", f)

        # Create product -> name mapping
        product_to_name = dict(zip(catalog_names['PRODUCT_ID'], catalog_names['NAME']))

        # Get unique products that appear in bids (relevant subset)
        bid_products = bids['PRODUCT_ID'].unique()
        log(f"Unique products in bids: {len(bid_products):,}", f)

        # Filter to products in catalog
        bid_products_with_name = [p for p in bid_products if p in product_to_name]
        log(f"Bid products with catalog NAME: {len(bid_products_with_name):,} ({len(bid_products_with_name)/len(bid_products)*100:.1f}%)", f)

        # Build TF-IDF on bid products only (for scalability)
        log("\nBuilding TF-IDF vectors...", f)
        product_list = list(bid_products_with_name)
        names_list = [product_to_name[p] for p in product_list]

        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        tfidf_matrix = vectorizer.fit_transform(names_list)

        log(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}", f)
        log(f"  TF-IDF matrix shape: {tfidf_matrix.shape}", f)
        log(f"  Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}", f)

        # Create product index mapping
        product_to_idx = {p: i for i, p in enumerate(product_list)}

        # ============================================================
        # 3. COMPUTE K PER AUCTION (from auction_panel)
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("COMPUTING SHOWN VS NOT-SHOWN SETS", f)
        log("=" * 80, f)

        # Get K per auction from auction_panel
        k_per_auction = auction_panel.groupby('auction_id')['K'].first().reset_index()
        k_dict = dict(zip(k_per_auction['auction_id'], k_per_auction['K']))
        log(f"\nAuctions with K values: {len(k_dict):,}", f)

        # Add K to bids and create shown indicator
        log("\nClassifying bids as shown (rank <= K) vs not-shown (rank > K)...", f)
        bids['K'] = bids['AUCTION_ID'].map(k_dict)
        bids = bids.dropna(subset=['K'])
        bids['shown'] = bids['RANKING'] <= bids['K']

        shown_bids = bids[bids['shown']].copy()
        not_shown_bids = bids[~bids['shown']].copy()

        log(f"  Shown bids (rank <= K): {len(shown_bids):,} ({len(shown_bids)/len(bids)*100:.1f}%)", f)
        log(f"  Not-shown bids (rank > K): {len(not_shown_bids):,} ({len(not_shown_bids)/len(bids)*100:.1f}%)", f)

        # ============================================================
        # 4. LINK AUCTIONS TO USERS VIA SESSIONS
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("LINKING AUCTIONS TO USER SESSIONS", f)
        log("=" * 80, f)

        # Get user-auction mapping
        auctions_users = auctions_users.rename(columns={
            'AUCTION_ID': 'auction_id',
            'OPAQUE_USER_ID': 'user_id',
            'CREATED_AT': 'auction_time'
        })
        auctions_users['auction_time'] = pd.to_datetime(auctions_users['auction_time'])

        log(f"\nUnique users in auctions: {auctions_users['user_id'].nunique():,}", f)

        # ============================================================
        # 5. FOR EACH PURCHASE, COMPUTE SIMILARITY TO SHOWN/NOT-SHOWN
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("COMPUTING HALO SIMILARITY SCORES", f)
        log("=" * 80, f)

        # Get purchases with product IDs that are in our TF-IDF index
        purchase_events = events[events['event_type'] == 'purchase'].copy()

        # Check what columns are in each dataframe
        log(f"\nEvents columns: {list(purchase_events.columns)}", f)
        log(f"Valid purchases columns: {list(valid_purchases.columns)}", f)

        # Use spend from events if available, otherwise from valid_purchases
        if 'spend' in purchase_events.columns:
            purchase_events = purchase_events.merge(
                valid_purchases[['user_id', 'product_id', 'click_vendor_id']],
                on=['user_id', 'product_id'],
                how='inner'
            )
        else:
            purchase_events = purchase_events.merge(
                valid_purchases[['user_id', 'product_id', 'spend', 'click_vendor_id']],
                on=['user_id', 'product_id'],
                how='inner'
            )
        log(f"\nPurchases with session info: {len(purchase_events):,}", f)
        log(f"Merged columns: {list(purchase_events.columns)}", f)

        # Filter to purchases where product is in TF-IDF index
        purchase_events['in_tfidf'] = purchase_events['product_id'].isin(product_to_idx)
        purchase_with_tfidf = purchase_events[purchase_events['in_tfidf']].copy()
        log(f"Purchases with TF-IDF embedding: {len(purchase_with_tfidf):,}", f)

        # Get user's auctions and their shown/not-shown products
        user_auction_map = auctions_users.groupby('user_id')['auction_id'].apply(list).to_dict()

        results = []

        log("\nComputing similarity for each purchase...", f)
        for idx, row in tqdm(purchase_with_tfidf.iterrows(), total=len(purchase_with_tfidf), desc="Purchases"):
            user_id = row['user_id']
            purchased_product = row['product_id']
            purchased_vendor = row.get('click_vendor_id', None)

            if user_id not in user_auction_map:
                continue

            user_auctions = user_auction_map[user_id]

            # Get shown and not-shown products for this user's auctions
            user_shown = shown_bids[shown_bids['AUCTION_ID'].isin(user_auctions)]
            user_not_shown = not_shown_bids[not_shown_bids['AUCTION_ID'].isin(user_auctions)]

            shown_products = user_shown['PRODUCT_ID'].unique()
            not_shown_products = user_not_shown['PRODUCT_ID'].unique()

            # Filter to products in TF-IDF
            shown_products = [p for p in shown_products if p in product_to_idx]
            not_shown_products = [p for p in not_shown_products if p in product_to_idx]

            if len(shown_products) == 0 and len(not_shown_products) == 0:
                continue

            # Get purchased product vector
            purchased_idx = product_to_idx.get(purchased_product)
            if purchased_idx is None:
                continue

            purchased_vec = tfidf_matrix[purchased_idx]

            # Compute similarity to shown products
            if len(shown_products) > 0:
                shown_indices = [product_to_idx[p] for p in shown_products]
                shown_vecs = tfidf_matrix[shown_indices]
                shown_sims = cosine_similarity(purchased_vec, shown_vecs).flatten()
                h_shown = shown_sims.max()
                h_shown_mean = shown_sims.mean()

                # Split by vendor
                shown_vendors = user_shown[user_shown['PRODUCT_ID'].isin(shown_products)].groupby('PRODUCT_ID')['VENDOR_ID'].first().to_dict()
                same_vendor_sims = [shown_sims[i] for i, p in enumerate(shown_products) if shown_vendors.get(p) == purchased_vendor]
                other_vendor_sims = [shown_sims[i] for i, p in enumerate(shown_products) if shown_vendors.get(p) != purchased_vendor]

                h_shown_same_vendor = max(same_vendor_sims) if same_vendor_sims else 0
                h_shown_other_vendor = max(other_vendor_sims) if other_vendor_sims else 0
            else:
                h_shown = 0
                h_shown_mean = 0
                h_shown_same_vendor = 0
                h_shown_other_vendor = 0

            # Compute similarity to not-shown products
            if len(not_shown_products) > 0:
                not_shown_indices = [product_to_idx[p] for p in not_shown_products]
                not_shown_vecs = tfidf_matrix[not_shown_indices]
                not_shown_sims = cosine_similarity(purchased_vec, not_shown_vecs).flatten()
                h_not_shown = not_shown_sims.max()
                h_not_shown_mean = not_shown_sims.mean()
            else:
                h_not_shown = 0
                h_not_shown_mean = 0

            results.append({
                'user_id': user_id,
                'product_id': purchased_product,
                'spend': row['spend'],
                'n_shown': len(shown_products),
                'n_not_shown': len(not_shown_products),
                'H_shown': h_shown,
                'H_shown_mean': h_shown_mean,
                'H_not_shown': h_not_shown,
                'H_not_shown_mean': h_not_shown_mean,
                'H_shown_same_vendor': h_shown_same_vendor,
                'H_shown_other_vendor': h_shown_other_vendor
            })

        results_df = pd.DataFrame(results)
        log(f"\nPurchases with similarity computed: {len(results_df):,}", f)

        # ============================================================
        # 6. ANALYSIS: SHOWN VS NOT-SHOWN COMPARISON
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("ANALYSIS: SHOWN VS NOT-SHOWN COMPARISON", f)
        log("=" * 80, f)

        if len(results_df) > 0:
            log("\n--- Summary Statistics ---", f)
            log(f"\nExposure set sizes:", f)
            log(f"  Mean shown products: {results_df['n_shown'].mean():.1f}", f)
            log(f"  Mean not-shown products: {results_df['n_not_shown'].mean():.1f}", f)

            log(f"\nSimilarity scores (max):", f)
            log(f"  H_shown (max):      mean={results_df['H_shown'].mean():.4f}, median={results_df['H_shown'].median():.4f}", f)
            log(f"  H_not_shown (max):  mean={results_df['H_not_shown'].mean():.4f}, median={results_df['H_not_shown'].median():.4f}", f)

            log(f"\nSimilarity scores (mean):", f)
            log(f"  H_shown_mean:      mean={results_df['H_shown_mean'].mean():.4f}", f)
            log(f"  H_not_shown_mean:  mean={results_df['H_not_shown_mean'].mean():.4f}", f)

            # Key test: Is H_shown > H_not_shown?
            log("\n--- Key Test: H_shown vs H_not_shown ---", f)
            results_df['H_diff'] = results_df['H_shown'] - results_df['H_not_shown']
            log(f"  Mean difference (H_shown - H_not_shown): {results_df['H_diff'].mean():.4f}", f)
            log(f"  % where H_shown > H_not_shown: {(results_df['H_diff'] > 0).mean()*100:.1f}%", f)

            # Paired t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(results_df['H_shown'], results_df['H_not_shown'])
            log(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}", f)

            if p_value < 0.05 and results_df['H_diff'].mean() > 0:
                log("  -> SIGNIFICANT: Shown ads are more similar to purchases than not-shown", f)
            elif p_value < 0.05:
                log("  -> SIGNIFICANT but negative: Not-shown more similar (unexpected)", f)
            else:
                log("  -> NOT SIGNIFICANT: No clear difference", f)

            # Vendor decomposition
            log("\n--- Vendor Decomposition ---", f)
            log(f"  H_shown_same_vendor:  mean={results_df['H_shown_same_vendor'].mean():.4f}", f)
            log(f"  H_shown_other_vendor: mean={results_df['H_shown_other_vendor'].mean():.4f}", f)

            # Where is the halo?
            same_stronger = (results_df['H_shown_same_vendor'] > results_df['H_shown_other_vendor']).mean()
            log(f"  % where same-vendor similarity > other-vendor: {same_stronger*100:.1f}%", f)

            # ============================================================
            # 7. LIFT ANALYSIS BY SIMILARITY DECILE
            # ============================================================
            log("\n--- Lift Analysis by Similarity Decile ---", f)

            results_df['H_shown_decile'] = pd.qcut(results_df['H_shown'], q=10, labels=False, duplicates='drop')

            lift_table = results_df.groupby('H_shown_decile').agg({
                'H_shown': 'mean',
                'spend': ['mean', 'sum', 'count']
            }).round(4)
            lift_table.columns = ['H_shown_mean', 'spend_mean', 'spend_total', 'n_purchases']
            log(str(lift_table), f)

            # ============================================================
            # 8. PLACEBO: COMPARE TO RANDOM SHUFFLE
            # ============================================================
            log("\n--- Placebo Test: Random Shuffle ---", f)

            # Shuffle product assignments and recompute
            np.random.seed(42)
            shuffled_indices = np.random.permutation(len(results_df))
            results_df['H_shown_shuffled'] = results_df['H_shown'].values[shuffled_indices]

            log(f"  Original H_shown mean: {results_df['H_shown'].mean():.4f}", f)
            log(f"  Shuffled H_shown mean: {results_df['H_shown_shuffled'].mean():.4f}", f)
            log(f"  (Should be similar if no structure)", f)

            # Save results
            results_df.to_parquet(DATA_DIR / 'semantic_halo_results.parquet', index=False)
            log(f"\nSaved results to {DATA_DIR / 'semantic_halo_results.parquet'}", f)

        else:
            log("\nNo purchases with similarity computed. Cannot run analysis.", f)

        # ============================================================
        # SUMMARY
        # ============================================================
        log("", f)
        log("=" * 80, f)
        log("SEMANTIC HALO SUMMARY", f)
        log("=" * 80, f)

        if len(results_df) > 0:
            log(f"\nPurchases analyzed: {len(results_df):,}", f)
            log(f"Mean H_shown: {results_df['H_shown'].mean():.4f}", f)
            log(f"Mean H_not_shown: {results_df['H_not_shown'].mean():.4f}", f)
            log(f"Mean difference: {results_df['H_diff'].mean():.4f}", f)

            if results_df['H_diff'].mean() > 0:
                log("\nINTERPRETATION:", f)
                log("Shown ads are more similar to purchased products than not-shown ads.", f)
                log("This suggests ads are influencing purchase decisions beyond intent matching.", f)
            else:
                log("\nINTERPRETATION:", f)
                log("Not-shown ads are equally or more similar to purchases.", f)
                log("Ads may not be driving inspiration; users find products independently.", f)

        log("", f)
        log("=" * 80, f)
        log("11_SEMANTIC_HALO COMPLETE", f)
        log("=" * 80, f)

if __name__ == "__main__":
    main()
