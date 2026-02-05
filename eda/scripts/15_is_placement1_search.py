#!/usr/bin/env python3
"""
Placement Behavioral Analysis

Raw metrics comparing placements across 24 behavioral dimensions:
1. Placement transitions and pogo-sticking patterns
2. Auction timing clusters
3. Vendor diversity per auction
4. First impression latency
5. Winner product overlap between consecutive auctions
6. Auction ID reuse patterns
7. CTR by rank (decay curves)
8. Vendor purity (top vendor share)
9. CTR by rank (hump detection)
10. Scroll depth (max rank reached)
11. Auction duration (time-in-auction)
12. Entry point analysis (user flow)
13. Winner payload size
14. Impression count patterns (even/odd)
15. Brand extraction from catalog
16. Product name noun variance
17. Session entry point
18. Bid volume per auction
19. Price variance
20. Click depth (average rank of clicks)
21. Rank 1 stability
22. PDP source attribution
23. Winner-to-bidder ratio
24. Brand in product name consistency
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "analysis/position-effects/0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def get_paths(round_name: str) -> dict:
    if round_name == "round1":
        return {
            "auctions_users": DATA_DIR / "round1/auctions_users_all.parquet",
            "auctions_results": DATA_DIR / "round1/auctions_results_all.parquet",
            "impressions": DATA_DIR / "round1/impressions_all.parquet",
            "clicks": DATA_DIR / "round1/clicks_all.parquet",
            "catalog": DATA_DIR / "round1/catalog_all.parquet",
        }
    raise ValueError(f"Unknown round: {round_name}")


def load_parquet(path, columns=None):
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path, columns=columns)


def test_pogo_sticking(au, clicks, f):
    """
    Test 1: Placement Transitions and Pogo-Sticking Patterns

    Measures: Transition frequencies, timing, and return-loop patterns by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 1: PLACEMENT TRANSITIONS AND POGO-STICKING", f)
    log("=" * 80, f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)
    au_sorted["prev_time"] = au_sorted.groupby("USER_ID")["CREATED_AT"].shift(1)
    au_sorted["time_gap"] = (au_sorted["CREATED_AT"] - au_sorted["prev_time"]).dt.total_seconds()

    clicks_by_auction = clicks.groupby("AUCTION_ID").size().rename("click_count")
    au_sorted = au_sorted.merge(clicks_by_auction, left_on="AUCTION_ID", right_index=True, how="left")
    au_sorted["click_count"] = au_sorted["click_count"].fillna(0)

    au_sorted["prev_had_click"] = au_sorted.groupby("USER_ID")["click_count"].shift(1) > 0

    transition_matrix = {}
    for (prev_p, curr_p), group in au_sorted.dropna(subset=["prev_placement"]).groupby(
        ["prev_placement", "PLACEMENT"]
    ):
        transition_matrix[(prev_p, curr_p)] = {
            "count": len(group),
            "count_within_60s": (group["time_gap"] <= 60).sum(),
            "count_with_prev_click": group["prev_had_click"].sum(),
            "count_within_60s_with_click": ((group["time_gap"] <= 60) & group["prev_had_click"]).sum(),
        }

    log("\nPlacement Transitions (all):", f)
    log(f"{'From':<6} {'To':<6} {'Count':>10} {'<60s':>10} {'%<60s':>8}", f)
    log("-" * 50, f)
    for (prev_p, curr_p), stats in sorted(transition_matrix.items()):
        pct = stats["count_within_60s"] / stats["count"] * 100 if stats["count"] > 0 else 0
        log(f"{prev_p:<6} {curr_p:<6} {stats['count']:>10,} {stats['count_within_60s']:>10,} {pct:>7.1f}%", f)

    log("\nPogo-sticking patterns (P_X -> P_Y -> P_X within 60s, with click in middle):", f)

    au_sorted["next_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(-1)
    au_sorted["next_time"] = au_sorted.groupby("USER_ID")["CREATED_AT"].shift(-1)
    au_sorted["next_time_gap"] = (au_sorted["next_time"] - au_sorted["CREATED_AT"]).dt.total_seconds()

    pogo_candidates = au_sorted[
        (au_sorted["prev_placement"] == au_sorted["next_placement"]) &
        (au_sorted["time_gap"] <= 60) &
        (au_sorted["next_time_gap"] <= 60) &
        (au_sorted["click_count"] > 0)
    ].copy()

    log(f"\nTotal pogo-stick patterns found: {len(pogo_candidates):,}", f)

    if len(pogo_candidates) > 0:
        pogo_by_pattern = pogo_candidates.groupby(
            ["prev_placement", "PLACEMENT"]
        ).size().sort_values(ascending=False)

        log(f"\n{'Pattern':<20} {'Count':>10}", f)
        log("-" * 35, f)
        for (orig, middle), cnt in pogo_by_pattern.items():
            log(f"P{int(orig)} -> P{int(middle)} -> P{int(orig)}    {cnt:>10,}", f)

    log("\nP1 -> P3 -> P1 specific analysis:", f)
    p1_p3_p1 = au_sorted[
        (au_sorted["prev_placement"] == 1) &
        (au_sorted["PLACEMENT"] == 3) &
        (au_sorted["next_placement"] == 1)
    ].copy()

    total_p1_to_p3 = len(au_sorted[
        (au_sorted["prev_placement"] == 1) & (au_sorted["PLACEMENT"] == 3)
    ])

    log(f"Total P1 -> P3 transitions: {total_p1_to_p3:,}", f)
    log(f"P1 -> P3 -> P1 loops (any timing): {len(p1_p3_p1):,}", f)

    if total_p1_to_p3 > 0:
        log(f"Loop rate: {len(p1_p3_p1) / total_p1_to_p3 * 100:.2f}%", f)

    fast_loops = p1_p3_p1[
        (p1_p3_p1["time_gap"] <= 60) & (p1_p3_p1["next_time_gap"] <= 60)
    ]
    log(f"Fast loops (both legs ≤60s): {len(fast_loops):,}", f)

    clicked_loops = fast_loops[fast_loops["click_count"] > 0]
    log(f"Fast loops with click on P3: {len(clicked_loops):,}", f)


def test_auction_clustering(au, f):
    """
    Test 2: Auction Timing Clusters

    Measures: Time gaps between consecutive auctions by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 2: AUCTION TIMING CLUSTERS", f)
    log("=" * 80, f)

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["time_gap"] = au_sorted.groupby("USER_ID")["CREATED_AT"].diff().dt.total_seconds()

    gaps = au_sorted[au_sorted["time_gap"] > 0].copy()

    log("\nTime gaps between consecutive auctions by placement:", f)
    log(f"{'Place':>6} {'N':>10} {'Mean':>8} {'Median':>8} {'<10s':>8} {'<30s':>8} {'<60s':>8} {'>300s':>8}", f)
    log("-" * 80, f)

    for placement in sorted(gaps["PLACEMENT"].unique()):
        subset = gaps[gaps["PLACEMENT"] == placement]["time_gap"]
        n = len(subset)
        mean = subset.mean()
        median = subset.median()
        pct_lt10 = (subset <= 10).mean() * 100
        pct_lt30 = (subset <= 30).mean() * 100
        pct_lt60 = (subset <= 60).mean() * 100
        pct_gt300 = (subset > 300).mean() * 100
        log(f"{placement:>6} {n:>10,} {mean:>8.1f} {median:>8.1f} {pct_lt10:>7.1f}% {pct_lt30:>7.1f}% {pct_lt60:>7.1f}% {pct_gt300:>7.1f}%", f)

    log("\nDetailed distribution by placement:", f)
    for placement in sorted(gaps["PLACEMENT"].unique()):
        subset = gaps[gaps["PLACEMENT"] == placement]["time_gap"]
        log(f"\nPlacement {placement}:", f)
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.1f}s", f)

    log("\nAuction clusters (2+ auctions within 30s) by placement:", f)
    au_sorted["cluster_break"] = au_sorted["time_gap"].fillna(999) > 30
    au_sorted["cluster_id"] = au_sorted.groupby("USER_ID")["cluster_break"].cumsum()

    cluster_sizes = au_sorted.groupby(["USER_ID", "cluster_id", "PLACEMENT"]).size().reset_index(name="cluster_size")
    multi_clusters = cluster_sizes[cluster_sizes["cluster_size"] >= 2]

    log(f"\n{'Place':>6} {'Clusters':>10} {'Mean Size':>10} {'Max Size':>10}", f)
    log("-" * 45, f)
    for placement in sorted(multi_clusters["PLACEMENT"].unique()):
        subset = multi_clusters[multi_clusters["PLACEMENT"] == placement]
        log(f"{placement:>6} {len(subset):>10,} {subset['cluster_size'].mean():>10.2f} {subset['cluster_size'].max():>10}", f)


def test_vendor_diversity(ar, au, f):
    """
    Test 3: Vendor Diversity per Auction

    Measures: Unique vendor count and concentration among winners by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 3: VENDOR DIVERSITY PER AUCTION", f)
    log("=" * 80, f)

    winners = ar[ar["IS_WINNER"] == True].copy()

    auction_diversity = winners.groupby("AUCTION_ID").agg(
        unique_vendors=("VENDOR_ID", "nunique"),
        total_winners=("VENDOR_ID", "count")
    ).reset_index()

    auction_diversity = auction_diversity.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nVendor diversity among winners by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Vendors':>12} {'Median':>8} {'Min':>6} {'Max':>6} {'StdDev':>8}", f)
    log("-" * 70, f)

    for placement in sorted(auction_diversity["PLACEMENT"].dropna().unique()):
        subset = auction_diversity[auction_diversity["PLACEMENT"] == placement]["unique_vendors"]
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>12.2f} {subset.median():>8.0f} {subset.min():>6} {subset.max():>6} {subset.std():>8.2f}", f)

    log("\nVendor diversity distribution by placement:", f)
    for placement in sorted(auction_diversity["PLACEMENT"].dropna().unique()):
        subset = auction_diversity[auction_diversity["PLACEMENT"] == placement]["unique_vendors"]
        log(f"\nPlacement {int(placement)}:", f)
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.0f} vendors", f)

    log("\nVendor concentration (top vendor share of winners) by placement:", f)
    top_vendor_share = winners.groupby(["AUCTION_ID", "VENDOR_ID"]).size().reset_index(name="count")
    total_per_auction = top_vendor_share.groupby("AUCTION_ID")["count"].sum().rename("total")
    top_vendor_share = top_vendor_share.merge(total_per_auction, on="AUCTION_ID")
    top_vendor_share["share"] = top_vendor_share["count"] / top_vendor_share["total"]

    max_share = top_vendor_share.groupby("AUCTION_ID")["share"].max().reset_index(name="top_vendor_share")
    max_share = max_share.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log(f"\n{'Place':>6} {'Mean Top Share':>15} {'Median':>10} {'>50%':>10} {'>75%':>10}", f)
    log("-" * 60, f)
    for placement in sorted(max_share["PLACEMENT"].dropna().unique()):
        subset = max_share[max_share["PLACEMENT"] == placement]["top_vendor_share"]
        pct_gt50 = (subset > 0.5).mean() * 100
        pct_gt75 = (subset > 0.75).mean() * 100
        log(f"{int(placement):>6} {subset.mean():>15.3f} {subset.median():>10.3f} {pct_gt50:>9.1f}% {pct_gt75:>9.1f}%", f)


def test_first_hit_latency(au, imp, f):
    """
    Test 4: First Impression Latency

    Measures: Time from auction creation to first impression by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 4: FIRST IMPRESSION LATENCY", f)
    log("=" * 80, f)

    if imp is None or len(imp) == 0:
        log("No impressions data available.", f)
        return

    first_imp = imp.groupby("AUCTION_ID")["OCCURRED_AT"].min().reset_index()
    first_imp.columns = ["AUCTION_ID", "FIRST_IMP"]

    merged = au.merge(first_imp, on="AUCTION_ID", how="inner")
    merged["latency"] = (merged["FIRST_IMP"] - merged["CREATED_AT"]).dt.total_seconds()

    log(f"\nAuctions with impressions: {len(merged):,} / {len(au):,} ({len(merged)/len(au)*100:.1f}%)", f)

    log("\nLatency by placement:", f)
    log(f"{'Place':>6} {'N':>10} {'Mean':>8} {'Median':>8} {'<1s':>8} {'<5s':>8} {'<0s':>8}", f)
    log("-" * 65, f)

    for placement in sorted(merged["PLACEMENT"].unique()):
        subset = merged[merged["PLACEMENT"] == placement]["latency"]
        n = len(subset)
        pct_lt1 = (subset < 1).mean() * 100
        pct_lt5 = (subset < 5).mean() * 100
        pct_neg = (subset < 0).mean() * 100
        log(f"{placement:>6} {n:>10,} {subset.mean():>8.2f} {subset.median():>8.2f} {pct_lt1:>7.1f}% {pct_lt5:>7.1f}% {pct_neg:>7.1f}%", f)

    log("\nLatency percentiles by placement:", f)
    for placement in sorted(merged["PLACEMENT"].unique()):
        subset = merged[merged["PLACEMENT"] == placement]["latency"]
        log(f"\nPlacement {placement}:", f)
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.2f}s", f)


def test_winner_overlap(ar, au, f):
    """
    Test 5: Winner Product Overlap

    Measures: Jaccard similarity of winner products between consecutive auctions.
    """
    log("\n" + "=" * 80, f)
    log("TEST 5: WINNER PRODUCT OVERLAP", f)
    log("=" * 80, f)

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winner_sets = winners.groupby("AUCTION_ID")["PRODUCT_ID"].apply(set).to_dict()

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["prev_auction"] = au_sorted.groupby("USER_ID")["AUCTION_ID"].shift(1)
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)

    consecutive = au_sorted.dropna(subset=["prev_auction"]).copy()

    log(f"\nCalculating Jaccard similarity for {len(consecutive):,} consecutive auction pairs...", f)

    def jaccard(a, b):
        if not isinstance(a, set) or not isinstance(b, set):
            return np.nan
        if len(a) == 0 or len(b) == 0:
            return np.nan
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0

    consecutive["curr_winners"] = consecutive["AUCTION_ID"].map(winner_sets)
    consecutive["prev_winners"] = consecutive["prev_auction"].map(winner_sets)

    jaccard_values = []
    for curr, prev in tqdm(
        zip(consecutive["curr_winners"], consecutive["prev_winners"]),
        total=len(consecutive),
        desc="Jaccard"
    ):
        jaccard_values.append(jaccard(curr, prev))

    consecutive["jaccard"] = jaccard_values

    valid_pairs = consecutive.dropna(subset=["jaccard"])

    log(f"\nValid pairs with both auctions having winners: {len(valid_pairs):,}", f)

    log("\nJaccard similarity by placement pair:", f)
    log(f"{'From':>6} {'To':>6} {'N':>10} {'Mean':>8} {'Median':>8} {'=0':>8} {'>0.5':>8} {'=1':>8}", f)
    log("-" * 75, f)

    for (prev_p, curr_p), group in valid_pairs.groupby(["prev_placement", "PLACEMENT"]):
        j = group["jaccard"]
        pct_zero = (j == 0).mean() * 100
        pct_gt50 = (j > 0.5).mean() * 100
        pct_one = (j == 1).mean() * 100
        log(f"{int(prev_p):>6} {int(curr_p):>6} {len(j):>10,} {j.mean():>8.3f} {j.median():>8.3f} {pct_zero:>7.1f}% {pct_gt50:>7.1f}% {pct_one:>7.1f}%", f)

    log("\nSame-placement consecutive auctions:", f)
    for placement in sorted(valid_pairs["PLACEMENT"].unique()):
        same_place = valid_pairs[
            (valid_pairs["PLACEMENT"] == placement) &
            (valid_pairs["prev_placement"] == placement)
        ]["jaccard"]
        if len(same_place) > 0:
            log(f"\nPlacement {int(placement)} -> {int(placement)} (N={len(same_place):,}):", f)
            for p in [10, 25, 50, 75, 90]:
                val = np.percentile(same_place, p)
                log(f"  P{p:02d}: {val:.3f}", f)


def test_auction_reuse(au, imp, f):
    """
    Test 6: Impression Span per Auction

    Measures: Time between first and last impression for same auction ID.
    """
    log("\n" + "=" * 80, f)
    log("TEST 6: IMPRESSION SPAN PER AUCTION", f)
    log("=" * 80, f)

    if imp is None or len(imp) == 0:
        log("No impressions data available.", f)
        return

    imp_times = imp.groupby("AUCTION_ID").agg(
        first_imp=("OCCURRED_AT", "min"),
        last_imp=("OCCURRED_AT", "max"),
        n_impressions=("INTERACTION_ID", "count")
    ).reset_index()

    imp_times["imp_span"] = (imp_times["last_imp"] - imp_times["first_imp"]).dt.total_seconds()
    imp_times = imp_times.merge(
        au[["AUCTION_ID", "PLACEMENT", "USER_ID", "CREATED_AT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nImpression span (last - first impression time) by placement:", f)
    log(f"{'Place':>6} {'N':>10} {'Mean Span':>10} {'Median':>10} {'>60s':>10} {'>300s':>10}", f)
    log("-" * 60, f)

    for placement in sorted(imp_times["PLACEMENT"].dropna().unique()):
        subset = imp_times[imp_times["PLACEMENT"] == placement]["imp_span"]
        pct_gt60 = (subset > 60).mean() * 100
        pct_gt300 = (subset > 300).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.1f} {pct_gt60:>9.1f}% {pct_gt300:>9.1f}%", f)

    log("\nLooking for P1 auctions with impressions, then P3 visit, then more P1 impressions...", f)

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()

    p1_auctions = au_sorted[au_sorted["PLACEMENT"] == 1][["USER_ID", "AUCTION_ID", "CREATED_AT"]].copy()
    p3_visits = au_sorted[au_sorted["PLACEMENT"] == 3][["USER_ID", "CREATED_AT"]].copy()
    p3_visits.columns = ["USER_ID", "P3_TIME"]

    p1_with_imps = p1_auctions.merge(
        imp_times[["AUCTION_ID", "first_imp", "last_imp", "imp_span"]],
        on="AUCTION_ID",
        how="inner"
    )

    extended_spans = p1_with_imps[p1_with_imps["imp_span"] > 30]
    log(f"\nP1 auctions with impression span > 30s: {len(extended_spans):,}", f)

    if len(extended_spans) > 0:
        reuse_candidates = 0
        for _, row in tqdm(extended_spans.iterrows(), total=len(extended_spans), desc="Checking reuse"):
            user_p3 = p3_visits[
                (p3_visits["USER_ID"] == row["USER_ID"]) &
                (p3_visits["P3_TIME"] > row["first_imp"]) &
                (p3_visits["P3_TIME"] < row["last_imp"])
            ]
            if len(user_p3) > 0:
                reuse_candidates += 1

        log(f"P1 auctions with P3 visit between first and last impression: {reuse_candidates:,}", f)


def test_ctr_decay(ar, clicks, au, f):
    """
    Test 7: CTR by Rank

    Measures: Click-through rate by rank position for each placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 7: CTR BY RANK", f)
    log("=" * 80, f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()

    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1

    winners = winners.merge(
        click_keys,
        on=["AUCTION_ID", "PRODUCT_ID"],
        how="left"
    )
    winners["clicked"] = winners["clicked"].fillna(0)

    log("\nCTR by rank and placement (ranks 1-20):", f)
    placements = sorted(winners["PLACEMENT"].dropna().unique())

    header = f"{'Rank':>6}"
    for placement in placements:
        header += f" {'P' + str(int(placement)) + ' CTR':>12} {'N':>10}"
    log(header, f)
    log("-" * (6 + 23 * len(placements)), f)

    for rank in range(1, 21):
        row = f"{rank:>6}"
        for placement in placements:
            subset = winners[
                (winners["PLACEMENT"] == placement) &
                (winners["RANKING"] == rank)
            ]
            if len(subset) > 0:
                ctr = subset["clicked"].mean() * 100
                row += f" {ctr:>11.3f}% {len(subset):>10,}"
            else:
                row += f" {'N/A':>12} {0:>10}"
        log(row, f)

    log("\nCTR decay ratios (rank 10 CTR / rank 1 CTR):", f)
    log(f"{'Place':>8} {'Rank1 CTR':>12} {'Rank10 CTR':>12} {'Ratio':>10}", f)
    log("-" * 50, f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        rank1 = winners[
            (winners["PLACEMENT"] == placement) & (winners["RANKING"] == 1)
        ]["clicked"].mean()
        rank10 = winners[
            (winners["PLACEMENT"] == placement) & (winners["RANKING"] == 10)
        ]["clicked"].mean()

        if rank1 > 0:
            ratio = rank10 / rank1
            log(f"{int(placement):>8} {rank1*100:>11.3f}% {rank10*100:>11.3f}% {ratio:>10.3f}", f)
        else:
            log(f"{int(placement):>8} {'N/A':>12} {'N/A':>12} {'N/A':>10}", f)

    log("\nCTR by rank buckets:", f)
    rank_buckets = pd.cut(
        winners["RANKING"],
        bins=[0, 3, 6, 10, 20, 40, 64],
        labels=["1-3", "4-6", "7-10", "11-20", "21-40", "41-64"]
    )
    winners["rank_bucket"] = rank_buckets

    ctr_by_bucket = winners.groupby(["PLACEMENT", "rank_bucket"], observed=False).agg(
        ctr=("clicked", "mean"),
        n=("clicked", "count")
    ).reset_index()

    log(f"\n{'Place':>8} {'Bucket':>10} {'CTR':>12} {'N':>12}", f)
    log("-" * 50, f)
    for _, row in ctr_by_bucket.sort_values(["PLACEMENT", "rank_bucket"]).iterrows():
        log(f"{int(row['PLACEMENT']):>8} {row['rank_bucket']:>10} {row['ctr']*100:>11.3f}% {row['n']:>12,}", f)


###############################################################################
# PART 2: ADDITIONAL BEHAVIORAL METRICS
###############################################################################


def test_p2_brand_purity(ar, au, f):
    """
    Test 8: Vendor Purity (Top Vendor Share)

    Measures: Share of winners from top vendor per auction by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 8: VENDOR PURITY (TOP VENDOR SHARE)", f)
    log("=" * 80, f)

    winners = ar[ar["IS_WINNER"] == True].copy()

    auction_stats = winners.groupby("AUCTION_ID").agg(
        unique_vendors=("VENDOR_ID", "nunique"),
        total_winners=("VENDOR_ID", "count"),
        top_vendor_count=("VENDOR_ID", lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0)
    ).reset_index()

    auction_stats["top_vendor_share"] = auction_stats["top_vendor_count"] / auction_stats["total_winners"]
    auction_stats["single_vendor"] = auction_stats["unique_vendors"] == 1

    auction_stats = auction_stats.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nVendor purity by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Single Vendor':>15} {'Top Share Mean':>15} {'Top Share >90%':>15}", f)
    log("-" * 75, f)

    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]
        single_pct = subset["single_vendor"].mean() * 100
        top_share_mean = subset["top_vendor_share"].mean()
        top_share_gt90 = (subset["top_vendor_share"] >= 0.9).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {single_pct:>14.1f}% {top_share_mean:>15.3f} {top_share_gt90:>14.1f}%", f)

    log("\nTop vendor share distribution by placement:", f)
    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]["top_vendor_share"]
        log(f"\nPlacement {int(placement)}:", f)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.3f} ({val*100:.1f}% of winners from top vendor)", f)

    log("\nUnique vendors per auction distribution:", f)
    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]["unique_vendors"]
        log(f"\nPlacement {int(placement)}:", f)
        log(f"  Mean: {subset.mean():.1f} vendors", f)
        log(f"  Median: {subset.median():.0f} vendors", f)
        log(f"  1 vendor: {(subset == 1).mean()*100:.1f}%", f)
        log(f"  2-5 vendors: {((subset >= 2) & (subset <= 5)).mean()*100:.1f}%", f)
        log(f"  6-10 vendors: {((subset >= 6) & (subset <= 10)).mean()*100:.1f}%", f)
        log(f"  >10 vendors: {(subset > 10).mean()*100:.1f}%", f)


def test_p2_discovery_hump(ar, clicks, au, f):
    """
    Test 9: CTR Shape Analysis

    Measures: CTR by rank with ratio comparisons (rank N vs rank 1).
    """
    log("\n" + "=" * 80, f)
    log("TEST 9: CTR SHAPE ANALYSIS", f)
    log("=" * 80, f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1
    winners = winners.merge(click_keys, on=["AUCTION_ID", "PRODUCT_ID"], how="left")
    winners["clicked"] = winners["clicked"].fillna(0)

    log("\nCTR by rank for each placement (ranks 1-15):", f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        log(f"\nPlacement {int(placement)}:", f)
        log(f"  {'Rank':>6} {'CTR':>10} {'N':>12} {'vs Rank 1':>12}", f)
        log("  " + "-" * 45, f)

        p_winners = winners[winners["PLACEMENT"] == placement]
        rank1_ctr = p_winners[p_winners["RANKING"] == 1]["clicked"].mean()

        for rank in range(1, 16):
            subset = p_winners[p_winners["RANKING"] == rank]
            if len(subset) > 0:
                ctr = subset["clicked"].mean() * 100
                if rank1_ctr > 0:
                    ratio = (subset["clicked"].mean() / rank1_ctr)
                    ratio_str = f"{ratio:.2f}x"
                else:
                    ratio_str = "N/A"
                log(f"  {rank:>6} {ctr:>9.3f}% {len(subset):>12,} {ratio_str:>12}", f)

    log("\nCTR comparison (rank 4-6 vs rank 1-2):", f)
    log(f"{'Place':>6} {'CTR 1-2':>12} {'CTR 4-6':>12} {'Ratio':>10}", f)
    log("-" * 45, f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        p_winners = winners[winners["PLACEMENT"] == placement]
        ctr_1_2 = p_winners[p_winners["RANKING"].isin([1, 2])]["clicked"].mean()
        ctr_4_6 = p_winners[p_winners["RANKING"].isin([4, 5, 6])]["clicked"].mean()

        if ctr_1_2 > 0:
            ratio = ctr_4_6 / ctr_1_2
        else:
            ratio = np.nan

        log(f"{int(placement):>6} {ctr_1_2*100:>11.3f}% {ctr_4_6*100:>11.3f}% {ratio:>10.3f}", f)


def test_p2_deep_scroll(ar, imp, au, f):
    """
    Test 10: Scroll Depth

    Measures: Maximum rank reached per auction by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 10: SCROLL DEPTH", f)
    log("=" * 80, f)

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "RANKING"]].copy()
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    max_rank_per_auction = winners.groupby(["AUCTION_ID", "PLACEMENT"])["RANKING"].max().reset_index()
    max_rank_per_auction.columns = ["AUCTION_ID", "PLACEMENT", "max_rank"]

    log("\nMaximum rank reached per auction by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Max':>10} {'Median':>10} {'>20':>10} {'>30':>10} {'>40':>10} {'>50':>10}", f)
    log("-" * 85, f)

    for placement in sorted(max_rank_per_auction["PLACEMENT"].dropna().unique()):
        subset = max_rank_per_auction[max_rank_per_auction["PLACEMENT"] == placement]["max_rank"]
        gt20 = (subset > 20).mean() * 100
        gt30 = (subset > 30).mean() * 100
        gt40 = (subset > 40).mean() * 100
        gt50 = (subset > 50).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.0f} {gt20:>9.1f}% {gt30:>9.1f}% {gt40:>9.1f}% {gt50:>9.1f}%", f)

    log("\nMax rank distribution by placement:", f)
    for placement in sorted(max_rank_per_auction["PLACEMENT"].dropna().unique()):
        subset = max_rank_per_auction[max_rank_per_auction["PLACEMENT"] == placement]["max_rank"]
        log(f"\nPlacement {int(placement)}:", f)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: rank {val:.0f}", f)

    if imp is not None and len(imp) > 0:
        log("\nImpression depth analysis (using impression counts):", f)
        imp_counts = imp.groupby("AUCTION_ID").size().reset_index(name="n_impressions")
        imp_counts = imp_counts.merge(
            au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
            on="AUCTION_ID",
            how="left"
        )

        log(f"\n{'Place':>6} {'Auctions':>10} {'Mean Imps':>10} {'Median':>10} {'>10':>10} {'>20':>10} {'>30':>10}", f)
        log("-" * 75, f)

        for placement in sorted(imp_counts["PLACEMENT"].dropna().unique()):
            subset = imp_counts[imp_counts["PLACEMENT"] == placement]["n_impressions"]
            gt10 = (subset > 10).mean() * 100
            gt20 = (subset > 20).mean() * 100
            gt30 = (subset > 30).mean() * 100
            log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.0f} {gt10:>9.1f}% {gt20:>9.1f}% {gt30:>9.1f}%", f)


def test_p2_auction_longevity(au, imp, f):
    """
    Test 11: Auction Duration

    Measures: Time between first and last impression per auction.
    """
    log("\n" + "=" * 80, f)
    log("TEST 11: AUCTION DURATION", f)
    log("=" * 80, f)

    if imp is None or len(imp) == 0:
        log("No impressions data available.", f)
        return

    imp_times = imp.groupby("AUCTION_ID").agg(
        first_imp=("OCCURRED_AT", "min"),
        last_imp=("OCCURRED_AT", "max"),
        n_impressions=("INTERACTION_ID", "count")
    ).reset_index()

    imp_times["duration_sec"] = (imp_times["last_imp"] - imp_times["first_imp"]).dt.total_seconds()
    imp_times = imp_times.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    multi_imp = imp_times[imp_times["n_impressions"] >= 2]

    log("\nAuction duration (first to last impression) for auctions with 2+ impressions:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean(s)':>10} {'Median(s)':>10} {'>30s':>10} {'>60s':>10} {'>120s':>10} {'>300s':>10}", f)
    log("-" * 95, f)

    for placement in sorted(multi_imp["PLACEMENT"].dropna().unique()):
        subset = multi_imp[multi_imp["PLACEMENT"] == placement]["duration_sec"]
        gt30 = (subset > 30).mean() * 100
        gt60 = (subset > 60).mean() * 100
        gt120 = (subset > 120).mean() * 100
        gt300 = (subset > 300).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.1f} {gt30:>9.1f}% {gt60:>9.1f}% {gt120:>9.1f}% {gt300:>9.1f}%", f)

    log("\nDuration percentiles by placement:", f)
    for placement in sorted(multi_imp["PLACEMENT"].dropna().unique()):
        subset = multi_imp[multi_imp["PLACEMENT"] == placement]["duration_sec"]
        log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.1f}s ({val/60:.1f} min)", f)


def test_p2_entry_point(au, clicks, f):
    """
    Test 12: Placement Entry Points

    Measures: Previous placement before each placement (user flow).
    """
    log("\n" + "=" * 80, f)
    log("TEST 12: PLACEMENT ENTRY POINTS", f)
    log("=" * 80, f)

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)
    au_sorted["prev_auction"] = au_sorted.groupby("USER_ID")["AUCTION_ID"].shift(1)
    au_sorted["time_gap"] = au_sorted.groupby("USER_ID")["CREATED_AT"].diff().dt.total_seconds()

    p2_entries = au_sorted[au_sorted["PLACEMENT"] == 2].copy()

    log(f"\nTotal P2 auctions: {len(p2_entries):,}", f)

    log("\nP2 entry sources (previous placement):", f)
    entry_sources = p2_entries["prev_placement"].value_counts(dropna=False)
    total_entries = len(p2_entries)

    log(f"{'Source':>15} {'Count':>12} {'Percentage':>12}", f)
    log("-" * 45, f)
    for source, count in entry_sources.items():
        source_str = f"P{int(source)}" if pd.notna(source) else "Session Start"
        log(f"{source_str:>15} {count:>12,} {count/total_entries*100:>11.1f}%", f)

    if clicks is not None and len(clicks) > 0:
        clicks_by_auction = clicks.groupby("AUCTION_ID").size().rename("click_count")
        p2_entries = p2_entries.merge(
            clicks_by_auction,
            left_on="prev_auction",
            right_index=True,
            how="left"
        )
        p2_entries["prev_had_click"] = p2_entries["click_count"].fillna(0) > 0

        log("\nP2 entries from P3 with click analysis:", f)
        p2_from_p3 = p2_entries[p2_entries["prev_placement"] == 3]
        log(f"  P2 entries from P3: {len(p2_from_p3):,}", f)

        if len(p2_from_p3) > 0:
            with_click = p2_from_p3["prev_had_click"].sum()
            log(f"  P3 had a click: {with_click:,} ({with_click/len(p2_from_p3)*100:.1f}%)", f)

            fast_entries = p2_from_p3[p2_from_p3["time_gap"] <= 30]
            log(f"  Fast transition (≤30s): {len(fast_entries):,} ({len(fast_entries)/len(p2_from_p3)*100:.1f}%)", f)

    log("\nTransition timing to P2 by source:", f)
    for source in sorted(p2_entries["prev_placement"].dropna().unique()):
        subset = p2_entries[p2_entries["prev_placement"] == source]["time_gap"]
        subset = subset[subset > 0]
        if len(subset) > 0:
            log(f"\nP{int(source)} -> P2 (N={len(subset):,}):", f)
            log(f"  Mean: {subset.mean():.1f}s, Median: {subset.median():.1f}s", f)
            log(f"  <10s: {(subset <= 10).mean()*100:.1f}%, <30s: {(subset <= 30).mean()*100:.1f}%", f)


def test_p2_winner_payload(ar, au, f):
    """
    Test 13: Winner Payload Size

    Measures: Number of winners per auction by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 13: WINNER PAYLOAD SIZE", f)
    log("=" * 80, f)

    winners = ar[ar["IS_WINNER"] == True].copy()
    winner_counts = winners.groupby("AUCTION_ID").size().reset_index(name="n_winners")
    winner_counts = winner_counts.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nWinner count statistics by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Min':>8} {'Max':>8} {'CV':>10}", f)
    log("-" * 85, f)

    for placement in sorted(winner_counts["PLACEMENT"].dropna().unique()):
        subset = winner_counts[winner_counts["PLACEMENT"] == placement]["n_winners"]
        cv = subset.std() / subset.mean() if subset.mean() > 0 else 0
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.0f} {subset.std():>10.1f} {subset.min():>8} {subset.max():>8} {cv:>10.3f}", f)

    log("\nWinner count distribution by placement:", f)
    for placement in sorted(winner_counts["PLACEMENT"].dropna().unique()):
        subset = winner_counts[winner_counts["PLACEMENT"] == placement]["n_winners"]
        log(f"\nPlacement {int(placement)}:", f)
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.0f} winners", f)

    log("\nWinner count consistency (% of auctions at system max):", f)
    for placement in sorted(winner_counts["PLACEMENT"].dropna().unique()):
        subset = winner_counts[winner_counts["PLACEMENT"] == placement]["n_winners"]
        mode_val = subset.mode().iloc[0] if len(subset.mode()) > 0 else 0
        at_mode = (subset == mode_val).mean() * 100
        at_40 = (subset == 40).mean() * 100
        at_48 = (subset == 48).mean() * 100
        at_56 = (subset == 56).mean() * 100
        at_64 = (subset == 64).mean() * 100
        log(f"\nPlacement {int(placement)}: mode={mode_val:.0f} ({at_mode:.1f}% at mode)", f)
        log(f"  =40: {at_40:.1f}%, =48: {at_48:.1f}%, =56: {at_56:.1f}%, =64: {at_64:.1f}%", f)


def test_p2_vertical_batching(imp, au, f):
    """
    Test 14: Impression Count Patterns

    Measures: Even vs odd impression counts per auction by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 14: IMPRESSION COUNT PATTERNS", f)
    log("=" * 80, f)

    if imp is None or len(imp) == 0:
        log("No impressions data available.", f)
        return

    imp_counts = imp.groupby("AUCTION_ID").size().reset_index(name="n_impressions")
    imp_counts = imp_counts.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    imp_counts["is_even"] = imp_counts["n_impressions"] % 2 == 0

    deep_auctions = imp_counts[imp_counts["n_impressions"] >= 10]

    log("\nEven vs Odd impression counts (auctions with 10+ impressions):", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Even Count':>12} {'Odd Count':>12} {'% Even':>10}", f)
    log("-" * 60, f)

    for placement in sorted(deep_auctions["PLACEMENT"].dropna().unique()):
        subset = deep_auctions[deep_auctions["PLACEMENT"] == placement]
        even_count = subset["is_even"].sum()
        odd_count = len(subset) - even_count
        even_pct = even_count / len(subset) * 100 if len(subset) > 0 else 0
        log(f"{int(placement):>6} {len(subset):>10,} {even_count:>12,} {odd_count:>12,} {even_pct:>9.1f}%", f)

    log("\nImpression count modulo patterns (auctions with 10+ impressions):", f)
    for placement in sorted(deep_auctions["PLACEMENT"].dropna().unique()):
        subset = deep_auctions[deep_auctions["PLACEMENT"] == placement]["n_impressions"]
        log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)

        mod2 = subset % 2
        mod4 = subset % 4

        log(f"  Mod 2 distribution: 0={((mod2==0).mean()*100):.1f}%, 1={((mod2==1).mean()*100):.1f}%", f)
        log(f"  Mod 4 distribution: 0={((mod4==0).mean()*100):.1f}%, 1={((mod4==1).mean()*100):.1f}%, 2={((mod4==2).mean()*100):.1f}%, 3={((mod4==3).mean()*100):.1f}%", f)

    log("\nAll auctions (including low-impression ones):", f)
    log(f"{'Place':>6} {'Auctions':>10} {'% Even (n>=2)':>15} {'% Even (n>=4)':>15} {'% Even (n>=10)':>15}", f)
    log("-" * 75, f)

    for placement in sorted(imp_counts["PLACEMENT"].dropna().unique()):
        subset = imp_counts[imp_counts["PLACEMENT"] == placement]
        ge2 = subset[subset["n_impressions"] >= 2]
        ge4 = subset[subset["n_impressions"] >= 4]
        ge10 = subset[subset["n_impressions"] >= 10]

        even_ge2 = ge2["is_even"].mean() * 100 if len(ge2) > 0 else 0
        even_ge4 = ge4["is_even"].mean() * 100 if len(ge4) > 0 else 0
        even_ge10 = ge10["is_even"].mean() * 100 if len(ge10) > 0 else 0

        log(f"{int(placement):>6} {len(subset):>10,} {even_ge2:>14.1f}% {even_ge4:>14.1f}% {even_ge10:>14.1f}%", f)


###############################################################################
# PART 3: TESTS 15-24 - CATALOG-BASED AND ADDITIONAL BEHAVIORAL METRICS
###############################################################################


def extract_brand_from_categories(categories):
    """Extract brand from CATEGORIES array field (pattern: brand#xyz)."""
    if categories is None or not isinstance(categories, (list, str)):
        return None
    if isinstance(categories, str):
        # Parse JSON-like string
        try:
            import json
            categories = json.loads(categories)
        except:
            return None
    for cat in categories:
        if isinstance(cat, str) and cat.startswith("brand#"):
            return cat.replace("brand#", "").strip()
    return None


def test_brand_from_catalog(ar, au, catalog, f):
    """
    Test 15: Brand Diversity from Catalog

    Measures: Brand diversity among auction winners using catalog CATEGORIES field.
    """
    log("\n" + "=" * 80, f)
    log("TEST 15: BRAND DIVERSITY FROM CATALOG", f)
    log("=" * 80, f)

    if catalog is None or len(catalog) == 0:
        log("No catalog data available.", f)
        return

    # Extract brands from catalog
    log("\nExtracting brands from catalog CATEGORIES field...", f)
    catalog["brand"] = catalog["CATEGORIES"].apply(extract_brand_from_categories)
    brands_available = catalog[catalog["brand"].notna()]
    log(f"  Products with brand info: {len(brands_available):,} / {len(catalog):,} ({len(brands_available)/len(catalog)*100:.1f}%)", f)

    # Get winners and join with catalog
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winners = winners.merge(catalog[["PRODUCT_ID", "brand"]], on="PRODUCT_ID", how="left")
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Brand diversity per auction
    auction_brands = winners.groupby("AUCTION_ID").agg(
        unique_brands=("brand", lambda x: x.dropna().nunique()),
        total_with_brand=("brand", lambda x: x.notna().sum()),
        total_winners=("brand", "count")
    ).reset_index()

    auction_brands = auction_brands.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nBrand diversity per auction by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Brands':>12} {'Median':>8} {'1 Brand':>10} {'>5 Brands':>10}", f)
    log("-" * 65, f)

    for placement in sorted(auction_brands["PLACEMENT"].dropna().unique()):
        subset = auction_brands[auction_brands["PLACEMENT"] == placement]
        # Only consider auctions with at least one brand
        with_brands = subset[subset["total_with_brand"] > 0]["unique_brands"]
        if len(with_brands) > 0:
            single_brand = (with_brands == 1).mean() * 100
            gt5_brands = (with_brands > 5).mean() * 100
            log(f"{int(placement):>6} {len(with_brands):>10,} {with_brands.mean():>12.2f} {with_brands.median():>8.0f} {single_brand:>9.1f}% {gt5_brands:>9.1f}%", f)

    # Top brand share per auction
    log("\nTop brand share (concentration) by placement:", f)
    top_brand_share = winners[winners["brand"].notna()].groupby(["AUCTION_ID", "brand"]).size().reset_index(name="count")
    total_per_auction = top_brand_share.groupby("AUCTION_ID")["count"].sum().rename("total")
    top_brand_share = top_brand_share.merge(total_per_auction, on="AUCTION_ID")
    top_brand_share["share"] = top_brand_share["count"] / top_brand_share["total"]

    max_brand_share = top_brand_share.groupby("AUCTION_ID")["share"].max().reset_index(name="top_brand_share")
    max_brand_share = max_brand_share.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log(f"\n{'Place':>6} {'Mean Top Share':>15} {'Median':>10} {'>50%':>10} {'>75%':>10} {'=100%':>10}", f)
    log("-" * 70, f)
    for placement in sorted(max_brand_share["PLACEMENT"].dropna().unique()):
        subset = max_brand_share[max_brand_share["PLACEMENT"] == placement]["top_brand_share"]
        pct_gt50 = (subset > 0.5).mean() * 100
        pct_gt75 = (subset > 0.75).mean() * 100
        pct_eq100 = (subset == 1.0).mean() * 100
        log(f"{int(placement):>6} {subset.mean():>15.3f} {subset.median():>10.3f} {pct_gt50:>9.1f}% {pct_gt75:>9.1f}% {pct_eq100:>9.1f}%", f)

    # Top brands by placement
    log("\nTop 10 brands by placement:", f)
    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        subset = winners[(winners["PLACEMENT"] == placement) & (winners["brand"].notna())]
        brand_counts = subset["brand"].value_counts().head(10)
        log(f"\nPlacement {int(placement)} (N={len(subset):,} winners with brand):", f)
        for brand, count in brand_counts.items():
            log(f"  {brand}: {count:,} ({count/len(subset)*100:.1f}%)", f)


def test_product_noun_variance(ar, au, catalog, f):
    """
    Test 16: Product Name Noun Variance

    Measures: Diversity of product types/nouns in winner product names.
    """
    log("\n" + "=" * 80, f)
    log("TEST 16: PRODUCT NAME NOUN VARIANCE", f)
    log("=" * 80, f)

    if catalog is None or len(catalog) == 0:
        log("No catalog data available.", f)
        return

    # Common product type nouns to look for
    product_nouns = [
        "bag", "purse", "tote", "handbag", "clutch", "backpack",
        "shoe", "shoes", "boot", "boots", "sneaker", "sneakers", "sandal", "sandals", "heel", "heels",
        "dress", "gown", "skirt", "blouse", "shirt", "top", "jacket", "coat", "sweater", "cardigan",
        "pants", "jeans", "shorts", "leggings", "trousers",
        "ring", "necklace", "bracelet", "earring", "earrings", "watch", "jewelry",
        "wallet", "belt", "scarf", "hat", "sunglasses", "glasses"
    ]

    def extract_nouns(name):
        """Extract product type nouns from name."""
        if not isinstance(name, str):
            return []
        name_lower = name.lower()
        found = []
        for noun in product_nouns:
            if noun in name_lower:
                found.append(noun)
        return found

    log("\nExtracting product nouns from catalog NAME field...", f)
    catalog["nouns"] = catalog["NAME"].apply(extract_nouns)
    catalog["n_nouns"] = catalog["nouns"].apply(len)
    catalog["primary_noun"] = catalog["nouns"].apply(lambda x: x[0] if len(x) > 0 else None)

    products_with_nouns = catalog[catalog["n_nouns"] > 0]
    log(f"  Products with identifiable nouns: {len(products_with_nouns):,} / {len(catalog):,} ({len(products_with_nouns)/len(catalog)*100:.1f}%)", f)

    # Get winners and join with catalog
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winners = winners.merge(catalog[["PRODUCT_ID", "primary_noun", "n_nouns"]], on="PRODUCT_ID", how="left")
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Noun diversity per auction
    auction_nouns = winners.groupby("AUCTION_ID").agg(
        unique_nouns=("primary_noun", lambda x: x.dropna().nunique()),
        total_with_noun=("primary_noun", lambda x: x.notna().sum())
    ).reset_index()

    auction_nouns = auction_nouns.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nNoun diversity per auction by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Nouns':>12} {'Median':>8} {'1 Noun':>10} {'>3 Nouns':>10}", f)
    log("-" * 65, f)

    for placement in sorted(auction_nouns["PLACEMENT"].dropna().unique()):
        subset = auction_nouns[auction_nouns["PLACEMENT"] == placement]
        with_nouns = subset[subset["total_with_noun"] > 0]["unique_nouns"]
        if len(with_nouns) > 0:
            single_noun = (with_nouns == 1).mean() * 100
            gt3_nouns = (with_nouns > 3).mean() * 100
            log(f"{int(placement):>6} {len(with_nouns):>10,} {with_nouns.mean():>12.2f} {with_nouns.median():>8.0f} {single_noun:>9.1f}% {gt3_nouns:>9.1f}%", f)

    # Top nouns by placement
    log("\nTop 10 product nouns by placement:", f)
    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        subset = winners[(winners["PLACEMENT"] == placement) & (winners["primary_noun"].notna())]
        noun_counts = subset["primary_noun"].value_counts().head(10)
        log(f"\nPlacement {int(placement)} (N={len(subset):,} winners with noun):", f)
        for noun, count in noun_counts.items():
            log(f"  {noun}: {count:,} ({count/len(subset)*100:.1f}%)", f)

    # Entropy calculation
    log("\nNoun entropy by placement (higher = more diverse):", f)
    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        subset = winners[(winners["PLACEMENT"] == placement) & (winners["primary_noun"].notna())]
        if len(subset) > 0:
            noun_counts = subset["primary_noun"].value_counts()
            probs = noun_counts / noun_counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            log(f"  Placement {int(placement)}: entropy = {entropy:.3f} bits (N={len(subset):,})", f)


def test_session_entry_point(au, f):
    """
    Test 17: Session Entry Point

    Measures: Which placement starts user sessions and transition patterns.
    """
    log("\n" + "=" * 80, f)
    log("TEST 17: SESSION ENTRY POINT", f)
    log("=" * 80, f)

    # Define session as: gap > 30 minutes = new session
    SESSION_GAP_SECONDS = 30 * 60

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["time_gap"] = au_sorted.groupby("USER_ID")["CREATED_AT"].diff().dt.total_seconds()
    au_sorted["is_session_start"] = (au_sorted["time_gap"].isna()) | (au_sorted["time_gap"] > SESSION_GAP_SECONDS)
    au_sorted["session_id"] = au_sorted.groupby("USER_ID")["is_session_start"].cumsum()

    # Session starts by placement
    session_starts = au_sorted[au_sorted["is_session_start"]]
    log(f"\nTotal sessions identified: {len(session_starts):,}", f)
    log(f"Session definition: gap > {SESSION_GAP_SECONDS/60:.0f} minutes", f)

    log("\nSession start placement distribution:", f)
    start_dist = session_starts["PLACEMENT"].value_counts().sort_index()
    log(f"{'Placement':>10} {'Sessions':>12} {'Percentage':>12}", f)
    log("-" * 40, f)
    for placement, count in start_dist.items():
        log(f"{int(placement):>10} {count:>12,} {count/len(session_starts)*100:>11.1f}%", f)

    # First transition from session start
    log("\nFirst transition from session start placement:", f)
    au_sorted["next_placement"] = au_sorted.groupby(["USER_ID", "session_id"])["PLACEMENT"].shift(-1)

    for start_placement in sorted(session_starts["PLACEMENT"].unique()):
        starts = session_starts[session_starts["PLACEMENT"] == start_placement]
        starts_with_next = starts.merge(
            au_sorted[["USER_ID", "session_id", "PLACEMENT", "next_placement"]].drop_duplicates(["USER_ID", "session_id"]),
            on=["USER_ID", "session_id", "PLACEMENT"],
            how="left"
        )

        log(f"\nSessions starting at P{int(start_placement)} (N={len(starts):,}):", f)
        next_dist = starts_with_next["next_placement"].value_counts(dropna=False)
        for next_p, count in next_dist.items():
            next_str = f"P{int(next_p)}" if pd.notna(next_p) else "No next"
            log(f"  -> {next_str}: {count:,} ({count/len(starts)*100:.1f}%)", f)

    # Sessions by length (number of auctions)
    session_lengths = au_sorted.groupby(["USER_ID", "session_id"]).size().reset_index(name="n_auctions")
    session_lengths = session_lengths.merge(
        session_starts[["USER_ID", "session_id", "PLACEMENT"]].rename(columns={"PLACEMENT": "start_placement"}),
        on=["USER_ID", "session_id"],
        how="left"
    )

    log("\nSession length by start placement:", f)
    log(f"{'Start':>8} {'Sessions':>12} {'Mean Len':>10} {'Median':>10} {'Single':>10}", f)
    log("-" * 55, f)
    for start_p in sorted(session_lengths["start_placement"].dropna().unique()):
        subset = session_lengths[session_lengths["start_placement"] == start_p]["n_auctions"]
        single_pct = (subset == 1).mean() * 100
        log(f"P{int(start_p):>7} {len(subset):>12,} {subset.mean():>10.1f} {subset.median():>10.0f} {single_pct:>9.1f}%", f)


def test_bid_volume(ar, au, f):
    """
    Test 18: Bid Volume per Auction

    Measures: Total number of bids (all rows in auctions_results) per auction.
    """
    log("\n" + "=" * 80, f)
    log("TEST 18: BID VOLUME PER AUCTION", f)
    log("=" * 80, f)

    # Count all bids (not just winners) per auction
    bid_counts = ar.groupby("AUCTION_ID").size().reset_index(name="n_bids")
    bid_counts = bid_counts.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nBid volume statistics by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Min':>8} {'Max':>8}", f)
    log("-" * 75, f)

    for placement in sorted(bid_counts["PLACEMENT"].dropna().unique()):
        subset = bid_counts[bid_counts["PLACEMENT"] == placement]["n_bids"]
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.0f} {subset.std():>10.1f} {subset.min():>8} {subset.max():>8}", f)

    log("\nBid volume distribution by placement:", f)
    for placement in sorted(bid_counts["PLACEMENT"].dropna().unique()):
        subset = bid_counts[bid_counts["PLACEMENT"] == placement]["n_bids"]
        log(f"\nPlacement {int(placement)}:", f)
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.0f} bids", f)

    # Bid volume buckets
    log("\nBid volume buckets by placement:", f)
    log(f"{'Place':>6} {'1-10':>10} {'11-30':>10} {'31-50':>10} {'51-100':>10} {'>100':>10}", f)
    log("-" * 60, f)

    for placement in sorted(bid_counts["PLACEMENT"].dropna().unique()):
        subset = bid_counts[bid_counts["PLACEMENT"] == placement]["n_bids"]
        b1_10 = ((subset >= 1) & (subset <= 10)).mean() * 100
        b11_30 = ((subset >= 11) & (subset <= 30)).mean() * 100
        b31_50 = ((subset >= 31) & (subset <= 50)).mean() * 100
        b51_100 = ((subset >= 51) & (subset <= 100)).mean() * 100
        b_gt100 = (subset > 100).mean() * 100
        log(f"{int(placement):>6} {b1_10:>9.1f}% {b11_30:>9.1f}% {b31_50:>9.1f}% {b51_100:>9.1f}% {b_gt100:>9.1f}%", f)


def test_price_variance(ar, au, catalog, f):
    """
    Test 19: Price Variance

    Measures: Price distribution and variance among auction winners.
    """
    log("\n" + "=" * 80, f)
    log("TEST 19: PRICE VARIANCE", f)
    log("=" * 80, f)

    if catalog is None or len(catalog) == 0:
        log("No catalog data available.", f)
        return

    # Get winners with catalog price
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winners = winners.merge(
        catalog[["PRODUCT_ID", "CATALOG_PRICE"]],
        on="PRODUCT_ID",
        how="left"
    )
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Filter to valid prices
    winners_with_price = winners[winners["CATALOG_PRICE"].notna() & (winners["CATALOG_PRICE"] > 0)]
    log(f"\nWinners with valid price: {len(winners_with_price):,} / {len(winners):,} ({len(winners_with_price)/len(winners)*100:.1f}%)", f)

    # Price statistics by placement
    log("\nPrice statistics by placement (catalog price):", f)
    log(f"{'Place':>6} {'N':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'CV':>10}", f)
    log("-" * 65, f)

    for placement in sorted(winners_with_price["PLACEMENT"].dropna().unique()):
        subset = winners_with_price[winners_with_price["PLACEMENT"] == placement]["CATALOG_PRICE"]
        cv = subset.std() / subset.mean() if subset.mean() > 0 else 0
        log(f"{int(placement):>6} {len(subset):>10,} ${subset.mean():>9.0f} ${subset.median():>9.0f} ${subset.std():>9.0f} {cv:>10.3f}", f)

    # Price variance per auction
    auction_price_stats = winners_with_price.groupby("AUCTION_ID").agg(
        mean_price=("CATALOG_PRICE", "mean"),
        std_price=("CATALOG_PRICE", "std"),
        min_price=("CATALOG_PRICE", "min"),
        max_price=("CATALOG_PRICE", "max"),
        n_products=("CATALOG_PRICE", "count")
    ).reset_index()

    auction_price_stats["cv"] = auction_price_stats["std_price"] / auction_price_stats["mean_price"]
    auction_price_stats["price_range"] = auction_price_stats["max_price"] - auction_price_stats["min_price"]

    auction_price_stats = auction_price_stats.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Filter to auctions with 2+ products
    multi_product = auction_price_stats[auction_price_stats["n_products"] >= 2]

    log("\nPer-auction price variance by placement (auctions with 2+ winners):", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean CV':>10} {'Median CV':>10} {'CV<0.5':>10} {'CV>1':>10}", f)
    log("-" * 65, f)

    for placement in sorted(multi_product["PLACEMENT"].dropna().unique()):
        subset = multi_product[multi_product["PLACEMENT"] == placement]["cv"].dropna()
        if len(subset) > 0:
            cv_lt05 = (subset < 0.5).mean() * 100
            cv_gt1 = (subset > 1).mean() * 100
            log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.3f} {subset.median():>10.3f} {cv_lt05:>9.1f}% {cv_gt1:>9.1f}%", f)

    # Price range per auction
    log("\nPrice range (max - min) per auction by placement:", f)
    log(f"{'Place':>6} {'Mean Range':>12} {'Median Range':>14} {'<$10':>10} {'>$100':>10}", f)
    log("-" * 60, f)

    for placement in sorted(multi_product["PLACEMENT"].dropna().unique()):
        subset = multi_product[multi_product["PLACEMENT"] == placement]["price_range"]
        lt10 = (subset < 10).mean() * 100
        gt100 = (subset > 100).mean() * 100
        log(f"{int(placement):>6} ${subset.mean():>11.0f} ${subset.median():>13.0f} {lt10:>9.1f}% {gt100:>9.1f}%", f)


def test_click_depth(ar, clicks, au, f):
    """
    Test 20: Click Depth (Average Rank of Clicks)

    Measures: What rank positions get clicked by placement.
    """
    log("\n" + "=" * 80, f)
    log("TEST 20: CLICK DEPTH (AVERAGE RANK OF CLICKS)", f)
    log("=" * 80, f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    # Get winners with rank
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Join clicks to get rank of clicked item
    clicked = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    clicked = clicked.merge(
        winners,
        on=["AUCTION_ID", "PRODUCT_ID"],
        how="inner"
    )

    log(f"\nClicks matched to winners: {len(clicked):,}", f)

    # Click rank statistics by placement
    log("\nClick rank statistics by placement:", f)
    log(f"{'Place':>6} {'Clicks':>10} {'Mean Rank':>10} {'Median':>10} {'StdDev':>10}", f)
    log("-" * 55, f)

    for placement in sorted(clicked["PLACEMENT"].dropna().unique()):
        subset = clicked[clicked["PLACEMENT"] == placement]["RANKING"]
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>10.1f} {subset.median():>10.0f} {subset.std():>10.1f}", f)

    # Click rank distribution
    log("\nClick rank distribution by placement:", f)
    log(f"{'Place':>6} {'1-3':>10} {'4-10':>10} {'11-20':>10} {'21+':>10}", f)
    log("-" * 50, f)

    for placement in sorted(clicked["PLACEMENT"].dropna().unique()):
        subset = clicked[clicked["PLACEMENT"] == placement]["RANKING"]
        r1_3 = ((subset >= 1) & (subset <= 3)).mean() * 100
        r4_10 = ((subset >= 4) & (subset <= 10)).mean() * 100
        r11_20 = ((subset >= 11) & (subset <= 20)).mean() * 100
        r21_plus = (subset >= 21).mean() * 100
        log(f"{int(placement):>6} {r1_3:>9.1f}% {r4_10:>9.1f}% {r11_20:>9.1f}% {r21_plus:>9.1f}%", f)

    # Click rank percentiles
    log("\nClick rank percentiles by placement:", f)
    for placement in sorted(clicked["PLACEMENT"].dropna().unique()):
        subset = clicked[clicked["PLACEMENT"] == placement]["RANKING"]
        log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)
        for p in [10, 25, 50, 75, 90, 95]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: rank {val:.0f}", f)

    # Rank 1 click rate
    log("\nRank 1 click concentration by placement:", f)
    for placement in sorted(clicked["PLACEMENT"].dropna().unique()):
        subset = clicked[clicked["PLACEMENT"] == placement]["RANKING"]
        rank1_pct = (subset == 1).mean() * 100
        rank1_2_pct = (subset <= 2).mean() * 100
        rank1_5_pct = (subset <= 5).mean() * 100
        log(f"  Placement {int(placement)}: Rank 1={rank1_pct:.1f}%, Rank 1-2={rank1_2_pct:.1f}%, Rank 1-5={rank1_5_pct:.1f}%", f)


def test_rank1_stability(ar, au, f):
    """
    Test 21: Rank 1 Stability

    Measures: How often the same product appears at Rank 1 for consecutive same-placement auctions.
    """
    log("\n" + "=" * 80, f)
    log("TEST 21: RANK 1 STABILITY", f)
    log("=" * 80, f)

    # Get Rank 1 products
    rank1 = ar[(ar["IS_WINNER"] == True) & (ar["RANKING"] == 1)][["AUCTION_ID", "PRODUCT_ID"]].copy()
    rank1.columns = ["AUCTION_ID", "RANK1_PRODUCT"]

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted = au_sorted.merge(rank1, on="AUCTION_ID", how="left")

    # For same-placement consecutive auctions, check if Rank 1 product matches
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)
    au_sorted["prev_rank1_product"] = au_sorted.groupby("USER_ID")["RANK1_PRODUCT"].shift(1)

    same_placement = au_sorted[
        (au_sorted["PLACEMENT"] == au_sorted["prev_placement"]) &
        (au_sorted["RANK1_PRODUCT"].notna()) &
        (au_sorted["prev_rank1_product"].notna())
    ].copy()

    same_placement["rank1_matches"] = same_placement["RANK1_PRODUCT"] == same_placement["prev_rank1_product"]

    log(f"\nConsecutive same-placement auctions with Rank 1 data: {len(same_placement):,}", f)

    log("\nRank 1 stability by placement:", f)
    log(f"{'Place':>6} {'Pairs':>12} {'Same Product':>14} {'Percentage':>12}", f)
    log("-" * 50, f)

    for placement in sorted(same_placement["PLACEMENT"].unique()):
        subset = same_placement[same_placement["PLACEMENT"] == placement]
        matches = subset["rank1_matches"].sum()
        pct = subset["rank1_matches"].mean() * 100
        log(f"{int(placement):>6} {len(subset):>12,} {matches:>14,} {pct:>11.1f}%", f)

    # Time-based stability analysis
    log("\nRank 1 stability by time gap (for same-placement pairs):", f)
    au_sorted["time_gap"] = au_sorted.groupby("USER_ID")["CREATED_AT"].diff().dt.total_seconds()
    same_placement = same_placement.merge(
        au_sorted[["USER_ID", "CREATED_AT", "time_gap"]].drop_duplicates(["USER_ID", "CREATED_AT"]),
        on=["USER_ID", "CREATED_AT"],
        how="left"
    )

    for placement in sorted(same_placement["PLACEMENT"].unique()):
        subset = same_placement[same_placement["PLACEMENT"] == placement]
        log(f"\nPlacement {int(placement)}:", f)

        for gap_max, label in [(10, "≤10s"), (30, "≤30s"), (60, "≤60s"), (300, "≤5min")]:
            gap_subset = subset[subset["time_gap"] <= gap_max]
            if len(gap_subset) > 0:
                stability_pct = gap_subset["rank1_matches"].mean() * 100
                log(f"  {label}: {stability_pct:.1f}% stable (N={len(gap_subset):,})", f)


def test_pdp_source(au, clicks, f):
    """
    Test 22: PDP Source Attribution

    Measures: Which placement drives clicks (and thus traffic to PDP/P3).
    """
    log("\n" + "=" * 80, f)
    log("TEST 22: PDP SOURCE ATTRIBUTION", f)
    log("=" * 80, f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    # Get placement for each click's auction
    click_placements = clicks[["AUCTION_ID"]].drop_duplicates().merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    click_counts_by_placement = click_placements["PLACEMENT"].value_counts().sort_index()
    total_clicks = len(click_placements)

    log("\nClicks by source placement:", f)
    log(f"{'Placement':>10} {'Clicks':>12} {'Percentage':>12}", f)
    log("-" * 40, f)
    for placement, count in click_counts_by_placement.items():
        log(f"P{int(placement):>9} {count:>12,} {count/total_clicks*100:>11.1f}%", f)

    # Clicks per auction by placement
    clicks_per_auction = clicks.groupby("AUCTION_ID").size().reset_index(name="n_clicks")
    clicks_per_auction = clicks_per_auction.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nClicks per auction by placement:", f)
    log(f"{'Place':>6} {'Auctions':>12} {'Mean Clicks':>12} {'Median':>10} {'Max':>8}", f)
    log("-" * 55, f)

    for placement in sorted(clicks_per_auction["PLACEMENT"].dropna().unique()):
        subset = clicks_per_auction[clicks_per_auction["PLACEMENT"] == placement]["n_clicks"]
        log(f"{int(placement):>6} {len(subset):>12,} {subset.mean():>12.2f} {subset.median():>10.0f} {subset.max():>8}", f)

    # Multi-click auctions
    log("\nMulti-click auction patterns:", f)
    log(f"{'Place':>6} {'1 click':>12} {'2 clicks':>12} {'3+ clicks':>12}", f)
    log("-" * 50, f)

    for placement in sorted(clicks_per_auction["PLACEMENT"].dropna().unique()):
        subset = clicks_per_auction[clicks_per_auction["PLACEMENT"] == placement]["n_clicks"]
        c1 = (subset == 1).mean() * 100
        c2 = (subset == 2).mean() * 100
        c3_plus = (subset >= 3).mean() * 100
        log(f"{int(placement):>6} {c1:>11.1f}% {c2:>11.1f}% {c3_plus:>11.1f}%", f)

    # Click rate (auctions with clicks / total auctions)
    log("\nClick rate by placement (auctions with at least 1 click):", f)
    auctions_with_clicks = clicks_per_auction["AUCTION_ID"].unique()
    total_auctions_by_placement = au.groupby("PLACEMENT")["AUCTION_ID"].nunique()

    clicks_by_placement = click_placements.groupby("PLACEMENT")["AUCTION_ID"].nunique()

    log(f"{'Place':>6} {'Total Auctions':>15} {'With Clicks':>12} {'Click Rate':>12}", f)
    log("-" * 50, f)

    for placement in sorted(total_auctions_by_placement.index):
        total = total_auctions_by_placement[placement]
        with_clicks = clicks_by_placement.get(placement, 0)
        rate = with_clicks / total * 100 if total > 0 else 0
        log(f"{int(placement):>6} {total:>15,} {with_clicks:>12,} {rate:>11.2f}%", f)


def test_winner_bidder_ratio(ar, au, f):
    """
    Test 23: Winner-to-Bidder Ratio

    Measures: Competition intensity (n_winners / n_total_bids per auction).
    """
    log("\n" + "=" * 80, f)
    log("TEST 23: WINNER-TO-BIDDER RATIO", f)
    log("=" * 80, f)

    # Count total bids and winners per auction
    auction_stats = ar.groupby("AUCTION_ID").agg(
        n_bids=("PRODUCT_ID", "count"),
        n_winners=("IS_WINNER", "sum")
    ).reset_index()

    auction_stats["winner_ratio"] = auction_stats["n_winners"] / auction_stats["n_bids"]
    auction_stats = auction_stats.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nWinner-to-bidder ratio by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Ratio':>12} {'Median':>10} {'<0.5':>10} {'>0.8':>10}", f)
    log("-" * 65, f)

    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]["winner_ratio"]
        lt05 = (subset < 0.5).mean() * 100
        gt08 = (subset > 0.8).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {subset.mean():>12.3f} {subset.median():>10.3f} {lt05:>9.1f}% {gt08:>9.1f}%", f)

    log("\nWinner ratio distribution by placement:", f)
    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]["winner_ratio"]
        log(f"\nPlacement {int(placement)}:", f)
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.3f} ({val*100:.1f}%)", f)

    # Competition intensity (lower ratio = more competition)
    log("\nCompetition intensity buckets by placement:", f)
    log(f"{'Place':>6} {'High (r<0.3)':>14} {'Med (0.3-0.7)':>14} {'Low (r>0.7)':>14}", f)
    log("-" * 55, f)

    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]["winner_ratio"]
        high = (subset < 0.3).mean() * 100
        med = ((subset >= 0.3) & (subset <= 0.7)).mean() * 100
        low = (subset > 0.7).mean() * 100
        log(f"{int(placement):>6} {high:>13.1f}% {med:>13.1f}% {low:>13.1f}%", f)

    # Bids and winners raw counts
    log("\nRaw bid and winner counts by placement:", f)
    log(f"{'Place':>6} {'Mean Bids':>12} {'Mean Winners':>14} {'Med Bids':>12} {'Med Winners':>14}", f)
    log("-" * 65, f)

    for placement in sorted(auction_stats["PLACEMENT"].dropna().unique()):
        subset = auction_stats[auction_stats["PLACEMENT"] == placement]
        log(f"{int(placement):>6} {subset['n_bids'].mean():>12.1f} {subset['n_winners'].mean():>14.1f} {subset['n_bids'].median():>12.0f} {subset['n_winners'].median():>14.0f}", f)


def test_brand_name_consistency(ar, au, catalog, f):
    """
    Test 24: Brand in Product Name Consistency

    Measures: How consistently brand names appear in product names within an auction.
    """
    log("\n" + "=" * 80, f)
    log("TEST 24: BRAND IN PRODUCT NAME CONSISTENCY", f)
    log("=" * 80, f)

    if catalog is None or len(catalog) == 0:
        log("No catalog data available.", f)
        return

    # Extract brand from categories
    catalog["brand"] = catalog["CATEGORIES"].apply(extract_brand_from_categories)

    # Check if brand appears in product name
    def brand_in_name(row):
        if pd.isna(row["brand"]) or pd.isna(row["NAME"]):
            return None
        brand = row["brand"].lower()
        name = row["NAME"].lower()
        return brand in name

    catalog["brand_in_name"] = catalog.apply(brand_in_name, axis=1)

    products_with_brand = catalog[catalog["brand"].notna()]
    brand_in_name_pct = products_with_brand["brand_in_name"].mean() * 100
    log(f"\nProducts where brand appears in name: {brand_in_name_pct:.1f}% (of {len(products_with_brand):,} with brand)", f)

    # Get winners and join with catalog
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winners = winners.merge(catalog[["PRODUCT_ID", "brand", "brand_in_name"]], on="PRODUCT_ID", how="left")
    winners = winners.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Per auction: % of winners with brand in name
    auction_brand_consistency = winners[winners["brand"].notna()].groupby("AUCTION_ID").agg(
        n_with_brand=("brand", "count"),
        n_brand_in_name=("brand_in_name", "sum")
    ).reset_index()

    auction_brand_consistency["brand_name_ratio"] = (
        auction_brand_consistency["n_brand_in_name"] / auction_brand_consistency["n_with_brand"]
    )
    auction_brand_consistency = auction_brand_consistency.merge(
        au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    log("\nBrand-in-name consistency by placement:", f)
    log(f"{'Place':>6} {'Auctions':>10} {'Mean Ratio':>12} {'All Match':>12} {'None Match':>12}", f)
    log("-" * 60, f)

    for placement in sorted(auction_brand_consistency["PLACEMENT"].dropna().unique()):
        subset = auction_brand_consistency[auction_brand_consistency["PLACEMENT"] == placement]
        ratio = subset["brand_name_ratio"]
        all_match = (ratio == 1.0).mean() * 100
        none_match = (ratio == 0.0).mean() * 100
        log(f"{int(placement):>6} {len(subset):>10,} {ratio.mean():>12.3f} {all_match:>11.1f}% {none_match:>11.1f}%", f)

    # Per auction: most common brand word in names
    log("\nBrand word frequency in product names by placement:", f)
    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        subset = winners[(winners["PLACEMENT"] == placement) & (winners["brand"].notna())]
        has_brand_in_name = subset[subset["brand_in_name"] == True]
        log(f"\nPlacement {int(placement)}: {len(has_brand_in_name):,}/{len(subset):,} winners have brand in name ({len(has_brand_in_name)/len(subset)*100 if len(subset)>0 else 0:.1f}%)", f)


def main():
    parser = argparse.ArgumentParser(description="Placement Identification EDA")
    parser.add_argument("--round", choices=["round1"], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"15_is_placement1_search_{args.round}.txt"

    with open(output_path, "w") as fh:
        log("=" * 80, fh)
        log("PLACEMENT BEHAVIORAL ANALYSIS", fh)
        log("=" * 80, fh)
        log(f"Round: {args.round}", fh)

        log("\nLoading data...", fh)

        au = load_parquet(
            paths["auctions_users"],
            columns=["AUCTION_ID", "USER_ID", "PLACEMENT", "CREATED_AT"]
        )
        au["CREATED_AT"] = pd.to_datetime(au["CREATED_AT"])
        au["PLACEMENT"] = au["PLACEMENT"].astype(int)
        log(f"  auctions_users: {len(au):,} rows", fh)

        ar = load_parquet(
            paths["auctions_results"],
            columns=["AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "IS_WINNER", "RANKING"]
        )
        log(f"  auctions_results: {len(ar):,} rows", fh)

        imp = load_parquet(
            paths["impressions"],
            columns=["INTERACTION_ID", "AUCTION_ID", "PRODUCT_ID", "OCCURRED_AT"]
        )
        if imp is not None:
            imp["OCCURRED_AT"] = pd.to_datetime(imp["OCCURRED_AT"])
            log(f"  impressions: {len(imp):,} rows", fh)
        else:
            log("  impressions: None", fh)

        clicks = load_parquet(
            paths["clicks"],
            columns=["INTERACTION_ID", "AUCTION_ID", "PRODUCT_ID", "OCCURRED_AT"]
        )
        if clicks is not None:
            clicks["OCCURRED_AT"] = pd.to_datetime(clicks["OCCURRED_AT"])
            log(f"  clicks: {len(clicks):,} rows", fh)
        else:
            log("  clicks: None", fh)

        catalog = load_parquet(
            paths["catalog"],
            columns=["PRODUCT_ID", "NAME", "CATALOG_PRICE", "CATEGORIES"]
        )
        if catalog is not None:
            log(f"  catalog: {len(catalog):,} rows", fh)
        else:
            log("  catalog: None", fh)

        log("\nPlacement distribution:", fh)
        for p, cnt in au["PLACEMENT"].value_counts().sort_index().items():
            pct = cnt / len(au) * 100
            log(f"  Placement {p}: {cnt:,} auctions ({pct:.1f}%)", fh)

        log("\n" + "#" * 80, fh)
        log("# TESTS 1-7: TRANSITIONS, TIMING, DIVERSITY, LATENCY, OVERLAP, CTR", fh)
        log("#" * 80, fh)

        test_pogo_sticking(au, clicks, fh)
        test_auction_clustering(au, fh)
        test_vendor_diversity(ar, au, fh)
        test_first_hit_latency(au, imp, fh)
        test_winner_overlap(ar, au, fh)
        test_auction_reuse(au, imp, fh)
        test_ctr_decay(ar, clicks, au, fh)

        log("\n" + "#" * 80, fh)
        log("# TESTS 8-14: PURITY, CTR SHAPE, DEPTH, DURATION, FLOW, PAYLOAD, BATCHING", fh)
        log("#" * 80, fh)

        test_p2_brand_purity(ar, au, fh)
        test_p2_discovery_hump(ar, clicks, au, fh)
        test_p2_deep_scroll(ar, imp, au, fh)
        test_p2_auction_longevity(au, imp, fh)
        test_p2_entry_point(au, clicks, fh)
        test_p2_winner_payload(ar, au, fh)
        test_p2_vertical_batching(imp, au, fh)

        log("\n" + "#" * 80, fh)
        log("# TESTS 15-24: CATALOG-BASED AND ADDITIONAL BEHAVIORAL METRICS", fh)
        log("#" * 80, fh)

        test_brand_from_catalog(ar, au, catalog, fh)
        test_product_noun_variance(ar, au, catalog, fh)
        test_session_entry_point(au, fh)
        test_bid_volume(ar, au, fh)
        test_price_variance(ar, au, catalog, fh)
        test_click_depth(ar, clicks, au, fh)
        test_rank1_stability(ar, au, fh)
        test_pdp_source(au, clicks, fh)
        test_winner_bidder_ratio(ar, au, fh)
        test_brand_name_consistency(ar, au, catalog, fh)

        log("\n" + "=" * 80, fh)
        log("EDA COMPLETE", fh)
        log("=" * 80, fh)
        log(f"\nOutput saved to: {output_path.resolve()}", fh)


if __name__ == "__main__":
    main()
