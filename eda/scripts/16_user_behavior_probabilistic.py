#!/usr/bin/env python3
"""
Probabilistic User Behavior Analysis

Metrics for understanding WHY users behave as they do:
1. Brand Lock-In Factor
2. Brand Capture Rate (Markov)
3. Return-to-Search Rate
4. Path Velocity (IPC)
5. Spoke Depth
6. Peak CTR Rank
7. Scroll Exhaustion Point
8. Impression Survivorship
9. Viewport Gaze Symmetry
10. Price Arbitrage Intent
11. Click-Price Elasticity
12. Cluster Click Density
13. Semantic Narrowing
14. Session Mode Classification
15. Banner Resistance
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "analysis/position-effects/0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

SESSION_GAP_SECONDS = 30 * 60


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


def extract_brand_from_categories(categories):
    """Extract brand from categories array (looks for 'brand#*' or 'brand-*' pattern)."""
    if categories is None:
        return None
    # Handle string representation of list
    if isinstance(categories, str):
        import ast
        try:
            categories = ast.literal_eval(categories)
        except (ValueError, SyntaxError):
            # Try simple parsing
            categories = [c.strip().strip('"').strip("'") for c in categories.strip("[]").split(",")]
    if not isinstance(categories, (list, np.ndarray)):
        return None
    for cat in categories:
        if isinstance(cat, str):
            cat = cat.strip().strip('"').strip("'")
            if cat.startswith("brand#"):
                return cat.replace("brand#", "")
            if cat.startswith("brand-"):
                return cat.replace("brand-", "")
    return None


def extract_nouns(name):
    """Extract simple nouns from product name (words > 3 chars, lowercase)."""
    if not isinstance(name, str):
        return set()
    words = re.findall(r'\b[a-zA-Z]{4,}\b', name.lower())
    stop_words = {'with', 'from', 'that', 'this', 'have', 'been', 'were', 'will',
                  'size', 'small', 'large', 'medium', 'good', 'great', 'best',
                  'color', 'style', 'condition', 'shipping', 'free', 'description'}
    return set(w for w in words if w not in stop_words)


def compute_jaccard(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return np.nan
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def compute_entropy(counts):
    """Compute Shannon entropy from counts."""
    if len(counts) == 0:
        return 0
    probs = np.array(counts) / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs + 1e-10))


def add_session_ids(au_sorted):
    """Add session IDs based on 30-minute gap."""
    au_sorted = au_sorted.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["time_gap"] = au_sorted.groupby("USER_ID")["CREATED_AT"].diff().dt.total_seconds()
    au_sorted["is_session_start"] = (au_sorted["time_gap"].isna()) | (au_sorted["time_gap"] > SESSION_GAP_SECONDS)
    au_sorted["session_id"] = au_sorted.groupby("USER_ID")["is_session_start"].cumsum()
    return au_sorted


# =============================================================================
# METRIC 1: Brand Lock-In Factor
# =============================================================================
def metric_brand_lockin(ar, au, clicks, catalog, f):
    """
    P(Brand_{n+1} | Click on Brand_n) - probability of brand loyalty after click.
    After clicking Brand X in P1/P2, measure % of next 5 auctions that are 100% Brand X winners.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 1: BRAND LOCK-IN FACTOR", f)
    log("=" * 80, f)
    log("Definition: P(Brand_{n+1} | Click on Brand_n)", f)
    log("After clicking Brand X, what % of next 5 auctions have 100% Brand X winners?", f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    # Get product brands from catalog
    log("\nExtracting brands from catalog...", f)
    catalog_brands = catalog[["PRODUCT_ID", "CATEGORIES"]].copy()
    catalog_brands["brand"] = catalog_brands["CATEGORIES"].apply(extract_brand_from_categories)
    brand_map = dict(zip(catalog_brands["PRODUCT_ID"], catalog_brands["brand"]))

    brands_found = sum(1 for v in brand_map.values() if v is not None)
    log(f"Products with brand: {brands_found:,} / {len(brand_map):,} ({brands_found/len(brand_map)*100:.1f}%)", f)

    # Get winners and their brands
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "VENDOR_ID"]].copy()
    winners["brand"] = winners["PRODUCT_ID"].map(brand_map)

    # Get clicks with auction info (clicks already has USER_ID, so only get PLACEMENT and CREATED_AT)
    clicks_aug = clicks.merge(
        au[["AUCTION_ID", "PLACEMENT", "CREATED_AT"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )
    clicks_aug["click_brand"] = clicks_aug["PRODUCT_ID"].map(brand_map)

    # Filter to clicks with known brands on P1/P2
    clicks_with_brand = clicks_aug[
        (clicks_aug["click_brand"].notna()) &
        (clicks_aug["PLACEMENT"].isin([1, 2]))
    ].copy()

    log(f"\nClicks with known brand on P1/P2: {len(clicks_with_brand):,}", f)

    # Sort auctions by user and time
    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["auction_seq"] = au_sorted.groupby("USER_ID").cumcount()

    # For each click, look at next 5 auctions
    results_by_placement = {1: [], 2: []}
    results_by_brand = {}

    for _, click_row in tqdm(clicks_with_brand.iterrows(), total=len(clicks_with_brand), desc="Checking lock-in"):
        user_id = click_row["USER_ID"]
        click_time = click_row["CREATED_AT"]
        click_brand = click_row["click_brand"]
        placement = click_row["PLACEMENT"]

        # Get next 5 auctions for this user after this click
        user_auctions = au_sorted[
            (au_sorted["USER_ID"] == user_id) &
            (au_sorted["CREATED_AT"] > click_time)
        ].head(5)

        if len(user_auctions) == 0:
            continue

        # Check if all winners in these auctions are the same brand
        auction_ids = user_auctions["AUCTION_ID"].tolist()
        next_winners = winners[winners["AUCTION_ID"].isin(auction_ids)]

        if len(next_winners) == 0:
            continue

        # Calculate % of winner slots with same brand
        same_brand_winners = (next_winners["brand"] == click_brand).sum()
        total_winners = len(next_winners)
        pct_same_brand = same_brand_winners / total_winners

        results_by_placement[placement].append(pct_same_brand)

        if click_brand not in results_by_brand:
            results_by_brand[click_brand] = []
        results_by_brand[click_brand].append(pct_same_brand)

    # Report results by placement
    log("\n--- Lock-In Rate by Source Placement ---", f)
    log(f"{'Placement':>10} {'N':>10} {'Mean %':>10} {'Median %':>10} {'>50%':>10} {'=100%':>10}", f)
    log("-" * 65, f)

    for placement in [1, 2]:
        values = results_by_placement[placement]
        if len(values) > 0:
            mean_pct = np.mean(values) * 100
            median_pct = np.median(values) * 100
            pct_gt50 = np.mean([v > 0.5 for v in values]) * 100
            pct_eq100 = np.mean([v == 1.0 for v in values]) * 100
            log(f"{'P' + str(placement):>10} {len(values):>10,} {mean_pct:>9.1f}% {median_pct:>9.1f}% {pct_gt50:>9.1f}% {pct_eq100:>9.1f}%", f)
        else:
            log(f"{'P' + str(placement):>10} {'N/A':>10}", f)

    # Report results by top brands
    log("\n--- Lock-In Rate by Brand (Top 15) ---", f)
    log(f"{'Brand':>25} {'N':>10} {'Mean %':>10} {'Median %':>10}", f)
    log("-" * 60, f)

    brand_stats = [(brand, len(vals), np.mean(vals)) for brand, vals in results_by_brand.items() if len(vals) >= 5]
    brand_stats.sort(key=lambda x: -x[1])

    for brand, n, mean_val in brand_stats[:15]:
        vals = results_by_brand[brand]
        median_val = np.median(vals)
        log(f"{brand[:25]:>25} {n:>10,} {mean_val*100:>9.1f}% {median_val*100:>9.1f}%", f)


# =============================================================================
# METRIC 2: Brand Capture Rate (Markov Transition)
# =============================================================================
def metric_brand_capture_rate(au, clicks, f):
    """
    P(P1 click -> P2 within 5 min) - search-to-brand-feed funnel.
    Markov transition matrix: user clicks in P1, next auction is P2.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 2: BRAND CAPTURE RATE (MARKOV TRANSITION)", f)
    log("=" * 80, f)
    log("Definition: After clicking in placement X, what is probability of next placement Y?", f)

    if clicks is None or len(clicks) == 0:
        log("No clicks data available.", f)
        return

    # Add click info to auctions
    clicks_by_auction = clicks.groupby("AUCTION_ID").size().rename("click_count")
    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted = au_sorted.merge(clicks_by_auction, left_on="AUCTION_ID", right_index=True, how="left")
    au_sorted["click_count"] = au_sorted["click_count"].fillna(0)
    au_sorted["had_click"] = au_sorted["click_count"] > 0

    # Get next placement
    au_sorted["next_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(-1)
    au_sorted["next_time"] = au_sorted.groupby("USER_ID")["CREATED_AT"].shift(-1)
    au_sorted["time_to_next"] = (au_sorted["next_time"] - au_sorted["CREATED_AT"]).dt.total_seconds()

    # Filter to transitions within 5 minutes
    transitions = au_sorted[
        (au_sorted["next_placement"].notna()) &
        (au_sorted["time_to_next"] <= 300) &
        (au_sorted["time_to_next"] > 0)
    ].copy()

    # Build Markov transition matrix WITH click context
    log("\n--- Transition Matrix WITHOUT Click (All Transitions) ---", f)

    all_placements = sorted(au_sorted["PLACEMENT"].dropna().unique())

    from_to_label = "From/To"
    header = f"\n{from_to_label:>8}"
    for p in all_placements:
        p_label = "P" + str(int(p))
        header += f" {p_label:>8}"
    header += f" {'N':>10}"
    log(header, f)
    log("-" * (8 + 9 * len(all_placements) + 10), f)

    for from_p in all_placements:
        subset = transitions[transitions["PLACEMENT"] == from_p]
        row_total = len(subset)
        from_label = "P" + str(int(from_p))
        row_str = f"{from_label:>8}"
        for to_p in all_placements:
            count = len(subset[subset["next_placement"] == to_p])
            pct = count / row_total * 100 if row_total > 0 else 0
            row_str += f" {pct:>7.1f}%"
        row_str += f" {row_total:>10,}"
        log(row_str, f)

    log("\n--- Transition Matrix WITH Click (Clicked Before Transition) ---", f)

    clicked_transitions = transitions[transitions["had_click"] == True]

    header = f"\n{from_to_label:>8}"
    for p in all_placements:
        p_label = "P" + str(int(p))
        header += f" {p_label:>8}"
    header += f" {'N':>10}"
    log(header, f)
    log("-" * (8 + 9 * len(all_placements) + 10), f)

    for from_p in all_placements:
        subset = clicked_transitions[clicked_transitions["PLACEMENT"] == from_p]
        row_total = len(subset)
        from_label = "P" + str(int(from_p))
        row_str = f"{from_label:>8}"
        for to_p in all_placements:
            count = len(subset[subset["next_placement"] == to_p])
            pct = count / row_total * 100 if row_total > 0 else 0
            row_str += f" {pct:>7.1f}%"
        row_str += f" {row_total:>10,}"
        log(row_str, f)

    # Specific P1 -> P2 capture rate
    log("\n--- P1 -> P2 Capture Rate (Search to Brand Feed) ---", f)

    p1_all = transitions[transitions["PLACEMENT"] == 1]
    p1_clicked = clicked_transitions[clicked_transitions["PLACEMENT"] == 1]

    p1_to_p2_all = len(p1_all[p1_all["next_placement"] == 2])
    p1_to_p2_clicked = len(p1_clicked[p1_clicked["next_placement"] == 2])

    log(f"P1 -> P2 (all): {p1_to_p2_all:,} / {len(p1_all):,} = {p1_to_p2_all/len(p1_all)*100:.2f}%", f) if len(p1_all) > 0 else None
    log(f"P1 -> P2 (clicked): {p1_to_p2_clicked:,} / {len(p1_clicked):,} = {p1_to_p2_clicked/len(p1_clicked)*100:.2f}%", f) if len(p1_clicked) > 0 else None


# =============================================================================
# METRIC 3: Return-to-Search (RTS) Rate
# =============================================================================
def metric_return_to_search(au, f):
    """
    % of P3 visits ending in P1 re-impression within 30s.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 3: RETURN-TO-SEARCH (RTS) RATE", f)
    log("=" * 80, f)
    log("Definition: % of P3 visits that return to P1 within 30 seconds", f)

    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["next_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(-1)
    au_sorted["next_time"] = au_sorted.groupby("USER_ID")["CREATED_AT"].shift(-1)
    au_sorted["time_to_next"] = (au_sorted["next_time"] - au_sorted["CREATED_AT"]).dt.total_seconds()

    # Also get previous placement to know source
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)

    # P3 visits
    p3_visits = au_sorted[au_sorted["PLACEMENT"] == 3].copy()

    log(f"\nTotal P3 visits: {len(p3_visits):,}", f)

    # P3 -> P1 within 30s
    p3_to_p1_30s = p3_visits[
        (p3_visits["next_placement"] == 1) &
        (p3_visits["time_to_next"] <= 30) &
        (p3_visits["time_to_next"] > 0)
    ]

    rts_rate = len(p3_to_p1_30s) / len(p3_visits) * 100 if len(p3_visits) > 0 else 0
    log(f"P3 -> P1 within 30s: {len(p3_to_p1_30s):,} ({rts_rate:.2f}%)", f)

    # Breakdown by source placement
    log("\n--- RTS Rate by Source Placement (what led to P3) ---", f)
    log(f"{'Source':>10} {'P3 Visits':>12} {'Return to P1':>15} {'RTS Rate':>12}", f)
    log("-" * 55, f)

    for source_p in sorted(p3_visits["prev_placement"].dropna().unique()):
        source_visits = p3_visits[p3_visits["prev_placement"] == source_p]
        returns = source_visits[
            (source_visits["next_placement"] == 1) &
            (source_visits["time_to_next"] <= 30) &
            (source_visits["time_to_next"] > 0)
        ]
        rate = len(returns) / len(source_visits) * 100 if len(source_visits) > 0 else 0
        log(f"{'P' + str(int(source_p)):>10} {len(source_visits):>12,} {len(returns):>15,} {rate:>11.2f}%", f)

    # Time distribution for P3 -> P1 returns
    log("\n--- Time to Return Distribution (P3 -> P1) ---", f)
    p3_to_p1_all = p3_visits[
        (p3_visits["next_placement"] == 1) &
        (p3_visits["time_to_next"] > 0)
    ]["time_to_next"]

    if len(p3_to_p1_all) > 0:
        percentiles = [10, 25, 50, 75, 90, 95]
        for p in percentiles:
            val = np.percentile(p3_to_p1_all, p)
            log(f"P{p:02d}: {val:.1f}s", f)


# =============================================================================
# METRIC 4: Path Velocity (Impressions per Click)
# =============================================================================
def metric_path_velocity(imp, clicks, au, f):
    """
    How many impressions before a click occurs.
    IPC = total_impressions / total_clicks per placement.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 4: PATH VELOCITY (IMPRESSIONS PER CLICK)", f)
    log("=" * 80, f)
    log("Definition: IPC = total_impressions / total_clicks (lower = higher intent)", f)

    if imp is None or clicks is None:
        log("Missing impressions or clicks data.", f)
        return

    # Add placement to impressions and clicks
    imp_aug = imp.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")
    clicks_aug = clicks.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    log("\n--- IPC by Placement ---", f)
    log(f"{'Placement':>12} {'Impressions':>15} {'Clicks':>12} {'IPC':>10} {'CTR':>10}", f)
    log("-" * 65, f)

    for placement in sorted(imp_aug["PLACEMENT"].dropna().unique()):
        n_imp = len(imp_aug[imp_aug["PLACEMENT"] == placement])
        n_clicks = len(clicks_aug[clicks_aug["PLACEMENT"] == placement])
        ipc = n_imp / n_clicks if n_clicks > 0 else np.inf
        ctr = n_clicks / n_imp * 100 if n_imp > 0 else 0
        log(f"{'P' + str(int(placement)):>12} {n_imp:>15,} {n_clicks:>12,} {ipc:>10.1f} {ctr:>9.2f}%", f)

    # IPC by user
    log("\n--- IPC Distribution by User ---", f)
    user_imp = imp_aug.groupby("USER_ID").size().rename("impressions")
    user_clicks = clicks_aug.groupby("USER_ID").size().rename("clicks")
    user_stats = pd.DataFrame({"impressions": user_imp}).join(pd.DataFrame({"clicks": user_clicks}), how="outer").fillna(0)
    user_stats["ipc"] = user_stats["impressions"] / user_stats["clicks"].replace(0, np.nan)

    valid_ipc = user_stats["ipc"].dropna()
    if len(valid_ipc) > 0:
        log(f"Users with clicks: {len(valid_ipc):,}", f)
        log(f"Mean IPC: {valid_ipc.mean():.1f}", f)
        log(f"Median IPC: {valid_ipc.median():.1f}", f)
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            log(f"P{p:02d}: {np.percentile(valid_ipc, p):.1f}", f)


# =============================================================================
# METRIC 5: Spoke Depth (Hub & Spoke)
# =============================================================================
def metric_spoke_depth(au, clicks, f):
    """
    How many P3 visits per P1 search before new search or session end.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 5: SPOKE DEPTH (HUB & SPOKE)", f)
    log("=" * 80, f)
    log("Definition: For each P1 auction, count subsequent P3 visits until next P1 or session end", f)

    au_sorted = add_session_ids(au)

    # Mark P1 as hubs
    au_sorted["is_p1"] = au_sorted["PLACEMENT"] == 1
    au_sorted["is_p3"] = au_sorted["PLACEMENT"] == 3

    # Create hub groups: each P1 starts a new hub
    au_sorted["hub_id"] = au_sorted.groupby(["USER_ID", "session_id"])["is_p1"].cumsum()

    # For each P1 hub, count P3 spokes
    hub_spoke_counts = []

    for (user_id, session_id, hub_id), group in tqdm(
        au_sorted.groupby(["USER_ID", "session_id", "hub_id"]),
        desc="Calculating spoke depths"
    ):
        if hub_id == 0:  # Before first P1 in session
            continue
        p3_count = group["is_p3"].sum()
        hub_spoke_counts.append({
            "user_id": user_id,
            "session_id": session_id,
            "hub_id": hub_id,
            "spoke_count": p3_count
        })

    if len(hub_spoke_counts) == 0:
        log("No P1 hubs found.", f)
        return

    spoke_df = pd.DataFrame(hub_spoke_counts)

    log(f"\nTotal P1 hubs: {len(spoke_df):,}", f)
    log(f"Total P3 spokes: {spoke_df['spoke_count'].sum():,}", f)
    log(f"Mean spokes per hub: {spoke_df['spoke_count'].mean():.2f}", f)
    log(f"Median spokes per hub: {spoke_df['spoke_count'].median():.0f}", f)

    log("\n--- Spoke Count Distribution ---", f)
    spoke_dist = spoke_df["spoke_count"].value_counts().sort_index()
    for count, freq in spoke_dist.head(15).items():
        pct = freq / len(spoke_df) * 100
        log(f"{count:>3} spokes: {freq:>8,} hubs ({pct:>5.1f}%)", f)

    log("\n--- Spoke Count Percentiles ---", f)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(spoke_df["spoke_count"], p)
        log(f"P{p:02d}: {val:.0f} spokes", f)


# =============================================================================
# METRIC 6: Peak CTR Rank (Warm-Up Effect)
# =============================================================================
def metric_peak_ctr_rank(ar, clicks, au, imp, f):
    """
    Which rank has peak CTR (not always rank 1).
    CTR by rank; find local maximum.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 6: PEAK CTR RANK (WARM-UP EFFECT)", f)
    log("=" * 80, f)
    log("Definition: Which rank has highest CTR? (not always rank 1)", f)

    if clicks is None or imp is None:
        log("Missing clicks or impressions data.", f)
        return

    # Get winners with placement
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    # Create click lookup
    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1

    # Merge clicks
    winners = winners.merge(click_keys, on=["AUCTION_ID", "PRODUCT_ID"], how="left")
    winners["clicked"] = winners["clicked"].fillna(0)

    log("\n--- CTR by Rank (All Placements) ---", f)
    log(f"{'Rank':>6} {'Winners':>12} {'Clicks':>10} {'CTR':>10}", f)
    log("-" * 45, f)

    rank_ctr = []
    for rank in range(1, 33):
        subset = winners[winners["RANKING"] == rank]
        n_winners = len(subset)
        n_clicks = subset["clicked"].sum()
        ctr = n_clicks / n_winners * 100 if n_winners > 0 else 0
        log(f"{rank:>6} {n_winners:>12,} {n_clicks:>10,} {ctr:>9.2f}%", f)
        rank_ctr.append({"rank": rank, "ctr": ctr, "n": n_winners})

    rank_ctr_df = pd.DataFrame(rank_ctr)
    if len(rank_ctr_df[rank_ctr_df["n"] > 100]) > 0:
        peak_rank = rank_ctr_df[rank_ctr_df["n"] > 100].sort_values("ctr", ascending=False).iloc[0]
        log(f"\nPeak CTR rank (n>100): Rank {int(peak_rank['rank'])} with {peak_rank['ctr']:.2f}% CTR", f)

    # By placement
    log("\n--- CTR by Rank by Placement (Ranks 1-15) ---", f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        log(f"\nPlacement {int(placement)}:", f)
        log(f"{'Rank':>6} {'Winners':>12} {'Clicks':>10} {'CTR':>10}", f)
        log("-" * 45, f)

        p_winners = winners[winners["PLACEMENT"] == placement]
        peak_ctr = 0
        peak_rank_p = 1

        for rank in range(1, 16):
            subset = p_winners[p_winners["RANKING"] == rank]
            n_winners = len(subset)
            n_clicks = subset["clicked"].sum()
            ctr = n_clicks / n_winners * 100 if n_winners > 0 else 0
            log(f"{rank:>6} {n_winners:>12,} {n_clicks:>10,} {ctr:>9.2f}%", f)

            if n_winners > 50 and ctr > peak_ctr:
                peak_ctr = ctr
                peak_rank_p = rank

        log(f"Peak rank: {peak_rank_p} ({peak_ctr:.2f}% CTR)", f)


# =============================================================================
# METRIC 7: Scroll Exhaustion Point
# =============================================================================
def metric_scroll_exhaustion(ar, clicks, au, f):
    """
    The "cliff" rank where CTR drops below 0.1% of Rank 1.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 7: SCROLL EXHAUSTION POINT", f)
    log("=" * 80, f)
    log("Definition: Rank where CTR drops below 0.1% of Rank 1 CTR", f)

    if clicks is None:
        log("No clicks data available.", f)
        return

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1
    winners = winners.merge(click_keys, on=["AUCTION_ID", "PRODUCT_ID"], how="left")
    winners["clicked"] = winners["clicked"].fillna(0)

    log("\n--- Exhaustion Point by Placement ---", f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        log(f"\nPlacement {int(placement)}:", f)

        p_winners = winners[winners["PLACEMENT"] == placement]

        # Get rank 1 CTR
        rank1 = p_winners[p_winners["RANKING"] == 1]
        ctr_rank1 = rank1["clicked"].sum() / len(rank1) * 100 if len(rank1) > 0 else 0
        threshold = ctr_rank1 * 0.001  # 0.1% of rank 1

        log(f"Rank 1 CTR: {ctr_rank1:.3f}%", f)
        log(f"Threshold (0.1% of R1): {threshold:.5f}%", f)

        # Find cliff rank
        cliff_rank = None
        for rank in range(2, 65):
            subset = p_winners[p_winners["RANKING"] == rank]
            if len(subset) < 10:
                continue
            ctr = subset["clicked"].sum() / len(subset) * 100
            if ctr < threshold and cliff_rank is None:
                cliff_rank = rank
                log(f"Cliff rank: {rank} (CTR: {ctr:.5f}%)", f)
                break

        if cliff_rank is None:
            log("No cliff rank found (CTR never dropped below threshold)", f)

    # Max rank reached per auction (P90)
    log("\n--- Max Rank Reached (Scroll Depth) by Placement ---", f)

    max_ranks = winners.groupby(["AUCTION_ID", "PLACEMENT"])["RANKING"].max().reset_index()

    for placement in sorted(max_ranks["PLACEMENT"].dropna().unique()):
        subset = max_ranks[max_ranks["PLACEMENT"] == placement]["RANKING"]
        log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(subset, p)
            log(f"  P{p:02d}: {val:.0f}", f)


# =============================================================================
# METRIC 8: Impression Survivorship (Kaplan-Meier)
# =============================================================================
def metric_impression_survivorship(ar, au, f):
    """
    At what rank do 50% of users stop scrolling?
    Survival curve: "Rank until Exit" - use max rank per auction.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 8: IMPRESSION SURVIVORSHIP (KAPLAN-MEIER STYLE)", f)
    log("=" * 80, f)
    log("Definition: P(user scrolls to at least rank K)", f)

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "RANKING"]].copy()
    winners = winners.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    # Max rank per auction
    max_ranks = winners.groupby(["AUCTION_ID", "PLACEMENT"])["RANKING"].max().reset_index()

    log("\n--- Survival Function by Placement ---", f)

    for placement in sorted(max_ranks["PLACEMENT"].dropna().unique()):
        log(f"\nPlacement {int(placement)}:", f)

        subset = max_ranks[max_ranks["PLACEMENT"] == placement]["RANKING"]
        n_auctions = len(subset)

        log(f"Total auctions: {n_auctions:,}", f)
        log(f"{'Rank':>6} {'Survival':>12} {'Cumulative Exit':>18}", f)
        log("-" * 40, f)

        median_rank = None
        quartile_25 = None
        quartile_75 = None

        for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
            survival = (subset >= k).mean() * 100
            cum_exit = 100 - survival
            log(f"{k:>6} {survival:>11.1f}% {cum_exit:>17.1f}%", f)

            if survival <= 75 and quartile_75 is None:
                quartile_75 = k
            if survival <= 50 and median_rank is None:
                median_rank = k
            if survival <= 25 and quartile_25 is None:
                quartile_25 = k

        log(f"\nMedian survival rank (50% drop-off): {median_rank if median_rank else '>50'}", f)
        log(f"Q1 survival rank (75% drop-off): {quartile_25 if quartile_25 else '>50'}", f)
        log(f"Q3 survival rank (25% drop-off): {quartile_75 if quartile_75 else '>50'}", f)


# =============================================================================
# METRIC 9: Viewport Gaze Symmetry (Odd/Even Skew)
# =============================================================================
def metric_gaze_symmetry(ar, clicks, au, f):
    """
    Click bias toward left (odd) vs right (even) column in 2-column UI.
    Ratio of clicks on odd ranks vs even ranks.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 9: VIEWPORT GAZE SYMMETRY (ODD/EVEN SKEW)", f)
    log("=" * 80, f)
    log("Definition: Click bias toward odd (left) vs even (right) ranks", f)
    log("Assumes 2-column UI where odd=left, even=right", f)

    if clicks is None:
        log("No clicks data available.", f)
        return

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1
    winners = winners.merge(click_keys, on=["AUCTION_ID", "PRODUCT_ID"], how="left")
    winners["clicked"] = winners["clicked"].fillna(0)

    winners["is_odd"] = winners["RANKING"] % 2 == 1
    winners["is_even"] = winners["RANKING"] % 2 == 0

    log("\n--- Gaze Symmetry by Placement ---", f)
    log(f"{'Placement':>12} {'Odd Clicks':>12} {'Even Clicks':>12} {'Odd/Even':>10} {'Chi-sq p':>12}", f)
    log("-" * 65, f)

    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        subset = winners[winners["PLACEMENT"] == placement]

        # Filter to comparable ranks (both odd and even have slots)
        max_rank = subset["RANKING"].max()
        comparable = subset[subset["RANKING"] <= max_rank]

        odd_clicks = comparable[comparable["is_odd"]]["clicked"].sum()
        even_clicks = comparable[comparable["is_even"]]["clicked"].sum()

        odd_total = len(comparable[comparable["is_odd"]])
        even_total = len(comparable[comparable["is_even"]])

        ratio = odd_clicks / even_clicks if even_clicks > 0 else np.inf

        # Chi-square test
        if odd_clicks + even_clicks > 0:
            expected_odd = (odd_clicks + even_clicks) * odd_total / (odd_total + even_total)
            expected_even = (odd_clicks + even_clicks) * even_total / (odd_total + even_total)
            chi_sq = ((odd_clicks - expected_odd)**2 / expected_odd +
                      (even_clicks - expected_even)**2 / expected_even)
            p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
        else:
            p_value = 1.0

        log(f"{'P' + str(int(placement)):>12} {odd_clicks:>12,} {even_clicks:>12,} {ratio:>10.2f} {p_value:>12.4f}", f)

    # Detailed by rank
    log("\n--- Click Distribution by Rank (First 20) ---", f)
    for placement in sorted(winners["PLACEMENT"].dropna().unique()):
        log(f"\nPlacement {int(placement)}:", f)
        log(f"{'Rank':>6} {'Impressions':>12} {'Clicks':>10} {'CTR':>10} {'Position':>10}", f)
        log("-" * 55, f)

        subset = winners[winners["PLACEMENT"] == placement]
        for rank in range(1, 21):
            r_sub = subset[subset["RANKING"] == rank]
            n = len(r_sub)
            clicks_n = r_sub["clicked"].sum()
            ctr = clicks_n / n * 100 if n > 0 else 0
            pos = "Left" if rank % 2 == 1 else "Right"
            log(f"{rank:>6} {n:>12,} {clicks_n:>10,} {ctr:>9.2f}% {pos:>10}", f)


# =============================================================================
# METRIC 10: Price Arbitrage Intent
# =============================================================================
def metric_price_arbitrage(ar, clicks, au, catalog, f):
    """
    Do users click cheaper items in P3 carousel?
    Compare clicked item price to main PDP item price.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 10: PRICE ARBITRAGE INTENT", f)
    log("=" * 80, f)
    log("Definition: Do users click cheaper products than the source product?", f)

    if clicks is None or catalog is None:
        log("Missing clicks or catalog data.", f)
        return

    # Get prices from catalog
    price_map = dict(zip(catalog["PRODUCT_ID"], catalog["CATALOG_PRICE"]))

    # Get P3 clicks
    clicks_aug = clicks.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")
    p3_clicks = clicks_aug[clicks_aug["PLACEMENT"] == 3].copy()

    if len(p3_clicks) == 0:
        log("No P3 clicks found.", f)
        return

    p3_clicks["click_price"] = p3_clicks["PRODUCT_ID"].map(price_map)

    # Get winners (potential source products)
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners["winner_price"] = winners["PRODUCT_ID"].map(price_map)

    # Get average price of top 3 winners per auction as "reference price"
    top_winners = winners[winners["RANKING"] <= 3].groupby("AUCTION_ID")["winner_price"].mean().rename("ref_price")

    p3_clicks = p3_clicks.merge(top_winners, on="AUCTION_ID", how="left")
    p3_clicks["price_delta"] = p3_clicks["click_price"] - p3_clicks["ref_price"]

    valid = p3_clicks.dropna(subset=["click_price", "ref_price"])

    log(f"\nP3 clicks with valid prices: {len(valid):,} / {len(p3_clicks):,}", f)

    if len(valid) > 0:
        log(f"\n--- Price Delta Statistics (click_price - ref_price) ---", f)
        log(f"Mean delta: ${valid['price_delta'].mean():.2f}", f)
        log(f"Median delta: ${valid['price_delta'].median():.2f}", f)

        pct_cheaper = (valid["price_delta"] < 0).mean() * 100
        pct_similar = ((valid["price_delta"] >= -5) & (valid["price_delta"] <= 5)).mean() * 100
        pct_expensive = (valid["price_delta"] > 0).mean() * 100

        log(f"\n% clicking cheaper (delta < 0): {pct_cheaper:.1f}%", f)
        log(f"% clicking similar (|delta| <= $5): {pct_similar:.1f}%", f)
        log(f"% clicking more expensive (delta > 0): {pct_expensive:.1f}%", f)

        log(f"\n--- Price Delta Percentiles ---", f)
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(valid["price_delta"], p)
            log(f"P{p:02d}: ${val:.2f}", f)


# =============================================================================
# METRIC 11: Click-Price Elasticity
# =============================================================================
def metric_click_price_elasticity(ar, clicks, au, catalog, f):
    """
    Z-score of clicked item price vs auction average.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 11: CLICK-PRICE ELASTICITY", f)
    log("=" * 80, f)
    log("Definition: Z-score of clicked price vs auction average price", f)

    if clicks is None or catalog is None:
        log("Missing clicks or catalog data.", f)
        return

    price_map = dict(zip(catalog["PRODUCT_ID"], catalog["CATALOG_PRICE"]))

    # Get winners with prices
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()
    winners["price"] = winners["PRODUCT_ID"].map(price_map)

    # Auction price stats
    auction_price_stats = winners.groupby("AUCTION_ID").agg(
        mean_price=("price", "mean"),
        std_price=("price", "std"),
        n_products=("price", "count")
    ).reset_index()

    # Get clicks with prices
    clicks_aug = clicks.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")
    clicks_aug["click_price"] = clicks_aug["PRODUCT_ID"].map(price_map)
    clicks_aug = clicks_aug.merge(auction_price_stats, on="AUCTION_ID", how="left")

    # Calculate Z-score
    clicks_aug["z_score"] = (clicks_aug["click_price"] - clicks_aug["mean_price"]) / clicks_aug["std_price"].replace(0, np.nan)

    valid = clicks_aug.dropna(subset=["z_score"])

    log(f"\nClicks with valid Z-scores: {len(valid):,} / {len(clicks_aug):,}", f)

    if len(valid) > 0:
        log(f"\n--- Z-Score Statistics (All Placements) ---", f)
        log(f"Mean Z-score: {valid['z_score'].mean():.3f}", f)
        log(f"Median Z-score: {valid['z_score'].median():.3f}", f)
        log(f"Std of Z-score: {valid['z_score'].std():.3f}", f)

        # Interpretation
        log(f"\n% clicking below average (Z < 0): {(valid['z_score'] < 0).mean()*100:.1f}%", f)
        log(f"% clicking cheap (Z < -1): {(valid['z_score'] < -1).mean()*100:.1f}%", f)
        log(f"% clicking expensive (Z > 1): {(valid['z_score'] > 1).mean()*100:.1f}%", f)

        log(f"\n--- Z-Score by Placement ---", f)
        for placement in sorted(valid["PLACEMENT"].dropna().unique()):
            subset = valid[valid["PLACEMENT"] == placement]["z_score"]
            log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)
            log(f"  Mean Z: {subset.mean():.3f}", f)
            log(f"  Median Z: {subset.median():.3f}", f)
            log(f"  % below avg: {(subset < 0).mean()*100:.1f}%", f)

        log(f"\n--- Z-Score Percentiles ---", f)
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(valid["z_score"], p)
            log(f"P{p:02d}: {val:.3f}", f)


# =============================================================================
# METRIC 12: Cluster Click Density
# =============================================================================
def metric_cluster_click_density(ar, clicks, au, f):
    """
    Are users comparing nearby items or spread-out items?
    Std dev of ranks of all clicked items within a single auction.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 12: CLUSTER CLICK DENSITY", f)
    log("=" * 80, f)
    log("Definition: Std dev of ranks of clicked items per auction", f)
    log("Low std = comparison shopping, High std = exploration", f)

    if clicks is None:
        log("No clicks data available.", f)
        return

    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID", "RANKING"]].copy()
    winners = winners.merge(au[["AUCTION_ID", "PLACEMENT"]].drop_duplicates(), on="AUCTION_ID", how="left")

    click_keys = clicks[["AUCTION_ID", "PRODUCT_ID"]].drop_duplicates()
    click_keys["clicked"] = 1
    winners = winners.merge(click_keys, on=["AUCTION_ID", "PRODUCT_ID"], how="left")
    winners["clicked"] = winners["clicked"].fillna(0)

    clicked_items = winners[winners["clicked"] == 1]

    # Group by auction
    auction_click_stats = clicked_items.groupby(["AUCTION_ID", "PLACEMENT"]).agg(
        n_clicks=("RANKING", "count"),
        mean_rank=("RANKING", "mean"),
        std_rank=("RANKING", "std"),
        min_rank=("RANKING", "min"),
        max_rank=("RANKING", "max")
    ).reset_index()

    # Multi-click auctions only for std
    multi_click = auction_click_stats[auction_click_stats["n_clicks"] >= 2]

    log(f"\nAuctions with clicks: {len(auction_click_stats):,}", f)
    log(f"Auctions with 2+ clicks: {len(multi_click):,}", f)

    if len(multi_click) > 0:
        log(f"\n--- Click Spread Statistics (2+ clicks per auction) ---", f)
        log(f"Mean std dev of ranks: {multi_click['std_rank'].mean():.2f}", f)
        log(f"Median std dev: {multi_click['std_rank'].median():.2f}", f)

        log(f"\n--- By Placement ---", f)
        for placement in sorted(multi_click["PLACEMENT"].dropna().unique()):
            subset = multi_click[multi_click["PLACEMENT"] == placement]
            log(f"\nPlacement {int(placement)} (N={len(subset):,}):", f)
            log(f"  Mean std dev: {subset['std_rank'].mean():.2f}", f)
            log(f"  Median std dev: {subset['std_rank'].median():.2f}", f)
            log(f"  Mean click count: {subset['n_clicks'].mean():.2f}", f)
            log(f"  Mean rank spread (max-min): {(subset['max_rank'] - subset['min_rank']).mean():.1f}", f)

        log(f"\n--- Std Dev Percentiles ---", f)
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(multi_click["std_rank"], p)
            log(f"P{p:02d}: {val:.2f}", f)


# =============================================================================
# METRIC 13: Semantic Narrowing (Query Refinement)
# =============================================================================
def metric_semantic_narrowing(ar, au, catalog, f):
    """
    Are users refining (same nouns, more adjectives) or pivoting (different nouns)?
    Jaccard similarity of nouns between consecutive auctions.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 13: SEMANTIC NARROWING (QUERY REFINEMENT)", f)
    log("=" * 80, f)
    log("Definition: Noun overlap between consecutive auctions", f)
    log("High overlap = refinement, Low overlap = pivot/boredom", f)

    if catalog is None:
        log("No catalog data available.", f)
        return

    # Get product nouns
    log("\nExtracting nouns from product names...", f)
    product_nouns = {}
    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Extracting nouns"):
        product_nouns[row["PRODUCT_ID"]] = extract_nouns(row["NAME"])

    # Get winners per auction
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()

    # Aggregate nouns per auction
    auction_nouns = {}
    for auction_id, group in tqdm(winners.groupby("AUCTION_ID"), desc="Aggregating nouns"):
        all_nouns = set()
        for pid in group["PRODUCT_ID"]:
            all_nouns.update(product_nouns.get(pid, set()))
        auction_nouns[auction_id] = all_nouns

    # Add to auctions
    au_sorted = au.sort_values(["USER_ID", "CREATED_AT"]).copy()
    au_sorted["prev_auction"] = au_sorted.groupby("USER_ID")["AUCTION_ID"].shift(1)
    au_sorted["prev_placement"] = au_sorted.groupby("USER_ID")["PLACEMENT"].shift(1)

    consecutive = au_sorted.dropna(subset=["prev_auction"]).copy()

    # Calculate Jaccard
    log(f"\nCalculating Jaccard for {len(consecutive):,} consecutive auctions...", f)

    jaccard_values = []
    for _, row in tqdm(consecutive.iterrows(), total=len(consecutive), desc="Jaccard"):
        curr_nouns = auction_nouns.get(row["AUCTION_ID"], set())
        prev_nouns = auction_nouns.get(row["prev_auction"], set())
        jaccard_values.append(compute_jaccard(curr_nouns, prev_nouns))

    consecutive["jaccard"] = jaccard_values
    valid = consecutive.dropna(subset=["jaccard"])

    log(f"Valid pairs: {len(valid):,}", f)

    if len(valid) > 0:
        log(f"\n--- Noun Overlap Statistics ---", f)
        log(f"Mean Jaccard: {valid['jaccard'].mean():.3f}", f)
        log(f"Median Jaccard: {valid['jaccard'].median():.3f}", f)

        # Refinement vs pivot rates
        high_overlap = (valid["jaccard"] > 0.5).mean() * 100
        moderate_overlap = ((valid["jaccard"] > 0.2) & (valid["jaccard"] <= 0.5)).mean() * 100
        low_overlap = (valid["jaccard"] <= 0.2).mean() * 100

        log(f"\nRefinement (Jaccard > 0.5): {high_overlap:.1f}%", f)
        log(f"Moderate (0.2 < Jaccard <= 0.5): {moderate_overlap:.1f}%", f)
        log(f"Pivot/Boredom (Jaccard <= 0.2): {low_overlap:.1f}%", f)

        log(f"\n--- By Placement Transition ---", f)
        for (prev_p, curr_p), group in valid.groupby(["prev_placement", "PLACEMENT"]):
            j = group["jaccard"]
            log(f"P{int(prev_p)} -> P{int(curr_p)}: mean={j.mean():.3f}, median={j.median():.3f}, n={len(j):,}", f)


# =============================================================================
# METRIC 14: Session Mode Classification (Noun Entropy)
# =============================================================================
def metric_session_mode(ar, au, catalog, f):
    """
    Low entropy = targeted shopping, high entropy = window shopping.
    Entropy of nouns across all products in a session.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 14: SESSION MODE CLASSIFICATION (NOUN ENTROPY)", f)
    log("=" * 80, f)
    log("Definition: Entropy of nouns across all winners in a session", f)
    log("Low entropy = targeted shopping, High entropy = window shopping", f)

    if catalog is None:
        log("No catalog data available.", f)
        return

    # Get product nouns
    product_nouns = {}
    for _, row in catalog.iterrows():
        product_nouns[row["PRODUCT_ID"]] = extract_nouns(row["NAME"])

    # Add session IDs
    au_sorted = add_session_ids(au)

    # Get winners
    winners = ar[ar["IS_WINNER"] == True][["AUCTION_ID", "PRODUCT_ID"]].copy()

    # Get session for each winner
    winners = winners.merge(
        au_sorted[["AUCTION_ID", "USER_ID", "session_id"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # Calculate entropy per session
    log("\nCalculating session entropy...", f)

    session_entropies = []
    for (user_id, session_id), group in tqdm(
        winners.dropna(subset=["session_id"]).groupby(["USER_ID", "session_id"]),
        desc="Session entropy"
    ):
        all_nouns = []
        for pid in group["PRODUCT_ID"]:
            all_nouns.extend(list(product_nouns.get(pid, set())))

        if len(all_nouns) == 0:
            continue

        noun_counts = list(Counter(all_nouns).values())
        entropy = compute_entropy(noun_counts)
        n_unique_nouns = len(set(all_nouns))

        session_entropies.append({
            "user_id": user_id,
            "session_id": session_id,
            "entropy": entropy,
            "n_products": len(group),
            "n_unique_nouns": n_unique_nouns
        })

    if len(session_entropies) == 0:
        log("No valid sessions found.", f)
        return

    entropy_df = pd.DataFrame(session_entropies)

    log(f"\nTotal sessions: {len(entropy_df):,}", f)

    log(f"\n--- Entropy Statistics ---", f)
    log(f"Mean entropy: {entropy_df['entropy'].mean():.3f}", f)
    log(f"Median entropy: {entropy_df['entropy'].median():.3f}", f)
    log(f"Std entropy: {entropy_df['entropy'].std():.3f}", f)

    log(f"\n--- Entropy Percentiles ---", f)
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(entropy_df["entropy"], p)
        log(f"P{p:02d}: {val:.3f}", f)

    # Mode classification
    low_entropy = entropy_df[entropy_df["entropy"] < entropy_df["entropy"].quantile(0.25)]
    high_entropy = entropy_df[entropy_df["entropy"] > entropy_df["entropy"].quantile(0.75)]

    log(f"\n--- Mode Classification ---", f)
    log(f"Targeted sessions (Q1 entropy): {len(low_entropy):,} ({len(low_entropy)/len(entropy_df)*100:.1f}%)", f)
    log(f"  Mean products: {low_entropy['n_products'].mean():.1f}", f)
    log(f"  Mean unique nouns: {low_entropy['n_unique_nouns'].mean():.1f}", f)

    log(f"\nDiscovery sessions (Q4 entropy): {len(high_entropy):,} ({len(high_entropy)/len(entropy_df)*100:.1f}%)", f)
    log(f"  Mean products: {high_entropy['n_products'].mean():.1f}", f)
    log(f"  Mean unique nouns: {high_entropy['n_unique_nouns'].mean():.1f}", f)


# =============================================================================
# METRIC 15: Banner Resistance (P5 Blindness)
# =============================================================================
def metric_banner_resistance(au, imp, f):
    """
    P5-to-P2 latency - are users scrolling past banners?
    Time between first P5 impression and first P2 impression in session.
    """
    log("\n" + "=" * 80, f)
    log("METRIC 15: BANNER RESISTANCE (P5 BLINDNESS)", f)
    log("=" * 80, f)
    log("Definition: Time between first P5 impression and first P2 impression", f)
    log("Very short latency suggests banner blindness (scrolling past)", f)

    if imp is None:
        log("No impressions data available.", f)
        return

    # Check if P5 exists
    au_placements = au["PLACEMENT"].unique()
    if 5 not in au_placements:
        log("P5 placement not found in data.", f)
        log(f"Available placements: {sorted(au_placements)}", f)
        return

    # Add session IDs
    au_sorted = add_session_ids(au)

    # Get impressions with placement and session (impressions already has USER_ID)
    imp_aug = imp.merge(
        au_sorted[["AUCTION_ID", "PLACEMENT", "session_id"]].drop_duplicates(),
        on="AUCTION_ID",
        how="left"
    )

    # First P5 and P2 per session
    p5_first = imp_aug[imp_aug["PLACEMENT"] == 5].groupby(
        ["USER_ID", "session_id"]
    )["OCCURRED_AT"].min().reset_index()
    p5_first.columns = ["USER_ID", "session_id", "first_p5"]

    p2_first = imp_aug[imp_aug["PLACEMENT"] == 2].groupby(
        ["USER_ID", "session_id"]
    )["OCCURRED_AT"].min().reset_index()
    p2_first.columns = ["USER_ID", "session_id", "first_p2"]

    # Merge
    combined = p5_first.merge(p2_first, on=["USER_ID", "session_id"], how="inner")
    combined["latency"] = (combined["first_p2"] - combined["first_p5"]).dt.total_seconds()

    log(f"\nSessions with both P5 and P2: {len(combined):,}", f)

    if len(combined) > 0:
        log(f"\n--- P5 -> P2 Latency Statistics ---", f)
        log(f"Mean latency: {combined['latency'].mean():.1f}s", f)
        log(f"Median latency: {combined['latency'].median():.1f}s", f)

        # Banner blindness indicator
        very_fast = (combined["latency"] < 0.5).mean() * 100
        fast = (combined["latency"] < 2).mean() * 100
        negative = (combined["latency"] < 0).mean() * 100

        log(f"\n% with latency < 0.5s (banner blindness): {very_fast:.1f}%", f)
        log(f"% with latency < 2s: {fast:.1f}%", f)
        log(f"% with negative latency (P2 before P5): {negative:.1f}%", f)

        log(f"\n--- Latency Percentiles ---", f)
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = np.percentile(combined["latency"], p)
            log(f"P{p:02d}: {val:.1f}s", f)
    else:
        log("No sessions with both P5 and P2 impressions found.", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Probabilistic User Behavior EDA")
    parser.add_argument("--round", choices=["round1"], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"16_user_behavior_probabilistic_{args.round}.txt"

    with open(output_path, "w") as fh:
        log(f"Probabilistic User Behavior Analysis - {args.round}", fh)
        log("=" * 80, fh)
        log(f"Output: {output_path}", fh)

        # Load data
        log("\n--- Loading Data ---", fh)

        au = load_parquet(
            paths["auctions_users"],
            columns=["AUCTION_ID", "USER_ID", "PLACEMENT", "CREATED_AT"]
        )
        log(f"auctions_users: {len(au):,} rows", fh)

        ar = load_parquet(
            paths["auctions_results"],
            columns=["AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "IS_WINNER", "RANKING"]
        )
        log(f"auctions_results: {len(ar):,} rows", fh)

        imp = load_parquet(
            paths["impressions"],
            columns=["INTERACTION_ID", "AUCTION_ID", "PRODUCT_ID", "USER_ID", "OCCURRED_AT"]
        )
        log(f"impressions: {len(imp) if imp is not None else 0:,} rows", fh)

        clicks = load_parquet(
            paths["clicks"],
            columns=["INTERACTION_ID", "AUCTION_ID", "PRODUCT_ID", "USER_ID", "OCCURRED_AT"]
        )
        log(f"clicks: {len(clicks) if clicks is not None else 0:,} rows", fh)

        catalog = load_parquet(
            paths["catalog"],
            columns=["PRODUCT_ID", "NAME", "CATALOG_PRICE", "CATEGORIES"]
        )
        log(f"catalog: {len(catalog) if catalog is not None else 0:,} rows", fh)

        # Convert timestamps
        if au is not None:
            au["CREATED_AT"] = pd.to_datetime(au["CREATED_AT"])
            # Convert PLACEMENT to int
            au["PLACEMENT"] = pd.to_numeric(au["PLACEMENT"], errors="coerce").astype("Int64")
        if imp is not None:
            imp["OCCURRED_AT"] = pd.to_datetime(imp["OCCURRED_AT"])
        if clicks is not None:
            clicks["OCCURRED_AT"] = pd.to_datetime(clicks["OCCURRED_AT"])

        # Run all metrics
        log("\n" + "#" * 80, fh)
        log("# PART 1: TRANSITION & LOCK-IN METRICS", fh)
        log("#" * 80, fh)

        metric_brand_lockin(ar, au, clicks, catalog, fh)
        metric_brand_capture_rate(au, clicks, fh)
        metric_return_to_search(au, fh)

        log("\n" + "#" * 80, fh)
        log("# PART 2: ENGAGEMENT EFFICIENCY METRICS", fh)
        log("#" * 80, fh)

        metric_path_velocity(imp, clicks, au, fh)
        metric_spoke_depth(au, clicks, fh)
        metric_peak_ctr_rank(ar, clicks, au, imp, fh)

        log("\n" + "#" * 80, fh)
        log("# PART 3: SCROLL & FATIGUE METRICS", fh)
        log("#" * 80, fh)

        metric_scroll_exhaustion(ar, clicks, au, fh)
        metric_impression_survivorship(ar, au, fh)
        metric_gaze_symmetry(ar, clicks, au, fh)

        log("\n" + "#" * 80, fh)
        log("# PART 4: PRICE & COMPARISON METRICS", fh)
        log("#" * 80, fh)

        metric_price_arbitrage(ar, clicks, au, catalog, fh)
        metric_click_price_elasticity(ar, clicks, au, catalog, fh)
        metric_cluster_click_density(ar, clicks, au, fh)

        log("\n" + "#" * 80, fh)
        log("# PART 5: SEMANTIC & INTENT METRICS", fh)
        log("#" * 80, fh)

        metric_semantic_narrowing(ar, au, catalog, fh)
        metric_session_mode(ar, au, catalog, fh)
        metric_banner_resistance(au, imp, fh)

        log("\n" + "=" * 80, fh)
        log("ANALYSIS COMPLETE", fh)
        log("=" * 80, fh)
        log(f"Output saved to: {output_path}", fh)


if __name__ == "__main__":
    main()
