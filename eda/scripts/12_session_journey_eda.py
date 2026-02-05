#!/usr/bin/env python3
"""
Session & Journey EDA

Consolidated deep-dive into how placements, ranks, and timelines shape clicks.
Sections:
    * Placement/session composition and CTRs
    * Rank vs. on-screen position exposure
    * Session length buckets & click propensity
    * Search -> impression -> click latencies
    * Impression gap / pagination diagnostics
    * Cross-placement transitions
    * Feature profiles (bids, qualities) by position & placement
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def get_paths(round_name: str) -> dict:
    if round_name == "round1":
        return {
            "auctions_users": DATA_DIR / "round1/auctions_users_all.parquet",
            "impressions": DATA_DIR / "round1/impressions_all.parquet",
            "clicks": DATA_DIR / "round1/clicks_all.parquet",
            "session_items": DATA_DIR / "round1/session_items.parquet",
        }
    if round_name == "round2":
        return {
            "auctions_users": DATA_DIR / "round2/auctions_users_r2.parquet",
            "impressions": DATA_DIR / "round2/impressions_r2.parquet",
            "clicks": DATA_DIR / "round2/clicks_r2.parquet",
        }
    raise ValueError(f"Unknown round: {round_name}")


def load_parquet(path, columns=None):
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path, columns=columns)


def describe_session_items(session_items: pd.DataFrame, fh):
    log("\n" + "=" * 80, fh)
    log("SESSION ITEMS OVERVIEW", fh)
    log("=" * 80, fh)
    log(f"Rows: {len(session_items):,}", fh)
    log(f"Unique sessions: {session_items['auction_id'].nunique():,}", fh)

    placement_counts = session_items["placement"].value_counts().sort_index()
    log("\nPlacement distribution:", fh)
    for placement, cnt in placement_counts.items():
        pct = cnt / len(session_items) * 100
        log(f"  Placement {placement}: {cnt:,} rows ({pct:.1f}%)", fh)

    ctr_by_place = (
        session_items.groupby("placement")["clicked"].mean().sort_index() * 100
    )
    log("\nCTR by placement:", fh)
    for placement, ctr in ctr_by_place.items():
        log(f"  Placement {placement}: {ctr:.2f}% CTR", fh)

    log("\nCTR by position (first 20 positions):", fh)
    ctr_pos = session_items.groupby("position")["clicked"].mean().reset_index()
    for _, row in ctr_pos[ctr_pos["position"] <= 20].iterrows():
        log(f"  Position {int(row['position']):>2}: {row['clicked']*100:>5.2f}%", fh)

    log("\nPlacement x position CTR (first 10 positions):", fh)
    for placement, subset in session_items.groupby("placement"):
        log(f"  Placement {placement}:", fh)
        small = subset.groupby("position")["clicked"].mean().reset_index()
        small = small[small["position"] <= 10]
        for _, row in small.iterrows():
            log(f"    Pos {int(row['position']):>2}: {row['clicked']*100:>5.2f}%", fh)

    session_lengths = (
        session_items[["auction_id", "n_items"]].drop_duplicates().set_index("auction_id")
    )
    counts = session_lengths["n_items"].value_counts().sort_index()
    log("\nSession lengths (first 30 values):", fh)
    for length, cnt in counts.head(30).items():
        pct = cnt / len(session_lengths) * 100
        log(f"  {int(length):>2} items: {cnt:>6,} sessions ({pct:>5.2f}%)", fh)

    buckets = pd.cut(
        session_lengths["n_items"],
        bins=[0, 2, 4, 8, 16, 32, 64, np.inf],
        labels=["1-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"],
    )
    clicks = session_items[["auction_id", "n_clicks"]].drop_duplicates().set_index(
        "auction_id"
    )
    combined = session_lengths.join(clicks, how="left")
    combined["bucket"] = buckets
    agg = (
        combined.groupby("bucket")
        .agg(
            sessions=("n_items", "count"),
            pct_with_click=("n_clicks", lambda x: (x > 0).mean() * 100),
            mean_clicks=("n_clicks", "mean"),
        )
        .dropna()
    )
    log("\nSession length buckets vs clicks:", fh)
    for bucket, row in agg.iterrows():
        log(
            f"  {bucket:>6}: sessions={int(row['sessions']):>6}, "
            f"%with_click={row['pct_with_click']:.2f}%, "
            f"mean_clicks={row['mean_clicks']:.3f}",
            fh,
        )


def rank_vs_position(session_items: pd.DataFrame, fh):
    log("\n" + "=" * 80, fh)
    log("RANK VS ON-SCREEN POSITION", fh)
    log("=" * 80, fh)
    diff = session_items["rank"] - session_items["position"]
    log("Rank - position stats:", fh)
    log(str(diff.describe()), fh)
    for percentile in [0, 25, 50, 75, 90, 95, 99, 100]:
        val = np.percentile(diff, percentile)
        log(f"  P{percentile:>2}: {val:.1f}", fh)

    rank_ctr = session_items.groupby("rank")["clicked"].mean().reset_index()
    log("\nCTR by rank (first 20 ranks):", fh)
    for _, row in rank_ctr[rank_ctr["rank"] <= 20].iterrows():
        log(f"  Rank {int(row['rank']):>2}: {row['clicked']*100:>5.2f}%", fh)

    tail = rank_ctr[rank_ctr["rank"] >= rank_ctr["rank"].max() - 10]
    log("\nCTR tail (highest ranks):", fh)
    for _, row in tail.iterrows():
        log(f"  Rank {int(row['rank']):>2}: {row['clicked']*100:>5.2f}%", fh)


def click_timing(impressions, clicks, fh):
    log("\n" + "=" * 80, fh)
    log("CLICK TIMING", fh)
    log("=" * 80, fh)
    if impressions is None or clicks is None or len(impressions) == 0:
        log("Skipping timing: impressions or clicks missing.", fh)
        return

    imps = impressions.copy()
    clk = clicks.copy()
    imps["OCCURRED_AT"] = pd.to_datetime(imps["OCCURRED_AT"])
    clk["OCCURRED_AT"] = pd.to_datetime(clk["OCCURRED_AT"])

    session_start = imps.groupby("AUCTION_ID")["OCCURRED_AT"].min().rename("session_start")
    clk = clk.merge(session_start, on="AUCTION_ID", how="left")
    clk["secs_from_session_start"] = (
        clk["OCCURRED_AT"] - clk["session_start"]
    ).dt.total_seconds()
    stats = clk["secs_from_session_start"].describe(percentiles=[0.1, 0.5, 0.9])
    log("Time from first impression to click:", fh)
    log(str(stats), fh)
    log(
        f"Negative durations: {(clk['secs_from_session_start'] < 0).mean()*100:.2f}%",
        fh,
    )

    first_imp = (
        imps.sort_values("OCCURRED_AT")
        .drop_duplicates(["AUCTION_ID", "PRODUCT_ID"])
        .rename(columns={"OCCURRED_AT": "FIRST_IMP"})
    )
    clk = clk.merge(
        first_imp[["AUCTION_ID", "PRODUCT_ID", "FIRST_IMP"]],
        on=["AUCTION_ID", "PRODUCT_ID"],
        how="left",
    )
    clk["secs_from_first_imp"] = (
        clk["OCCURRED_AT"] - clk["FIRST_IMP"]
    ).dt.total_seconds()
    stats = clk["secs_from_first_imp"].describe(percentiles=[0.1, 0.5, 0.9])
    log("\nTime from product impression to click:", fh)
    log(str(stats), fh)
    log(
        f"Negative durations: {(clk['secs_from_first_imp'] < 0).mean()*100:.2f}%",
        fh,
    )


def search_to_impression(auctions_users, impressions, fh):
    log("\n" + "=" * 80, fh)
    log("SEARCH -> FIRST IMPRESSION LATENCY", fh)
    log("=" * 80, fh)
    if auctions_users is None or impressions is None:
        log("Skipping: auctions_users or impressions missing.", fh)
        return

    au = auctions_users.copy()
    au["CREATED_AT"] = pd.to_datetime(au["CREATED_AT"])
    imps = impressions[["AUCTION_ID", "OCCURRED_AT"]].copy()
    imps["OCCURRED_AT"] = pd.to_datetime(imps["OCCURRED_AT"])
    first_imp = imps.groupby("AUCTION_ID")["OCCURRED_AT"].min().rename("FIRST_IMP")
    merged = au.merge(first_imp, on="AUCTION_ID", how="left")
    merged["search_to_imp"] = (
        merged["FIRST_IMP"] - merged["CREATED_AT"]
    ).dt.total_seconds()

    series = merged["search_to_imp"].dropna()
    log(f"Latency count: {len(series):,}", fh)
    log(str(series.describe(percentiles=[0.1, 0.5, 0.9])), fh)
    log(f"Negative latencies: {(series < 0).mean()*100:.2f}%", fh)

    log("\nBy placement:", fh)
    for placement, subset in merged.groupby("PLACEMENT"):
        vals = subset["search_to_imp"].dropna()
        if len(vals) == 0:
            continue
        log(
            f"  Placement {placement}: median={vals.median():.2f}s, "
            f"P90={vals.quantile(0.9):.2f}s, mean={vals.mean():.2f}s, "
            f"neg%={(vals < 0).mean()*100:.2f}",
            fh,
        )


def pagination_gaps(auctions_users, impressions, fh):
    log("\n" + "=" * 80, fh)
    log("IMPRESSION GAPS / PAGINATION", fh)
    log("=" * 80, fh)
    if impressions is None:
        log("Skipping: impressions missing.", fh)
        return

    imps = impressions.copy()
    imps["OCCURRED_AT"] = pd.to_datetime(imps["OCCURRED_AT"])
    imps = imps.sort_values(["AUCTION_ID", "OCCURRED_AT"])
    imps["gap_secs"] = imps.groupby("AUCTION_ID")["OCCURRED_AT"].diff().dt.total_seconds()
    positive = imps["gap_secs"].dropna()
    positive = positive[positive > 0]
    if len(positive) == 0:
        log("No positive gaps.", fh)
        return

    log("Overall positive gaps:", fh)
    log(str(positive.describe(percentiles=[0.5, 0.9, 0.99])), fh)
    log(
        f">10s={(positive > 10).mean()*100:.2f}% | >30s={(positive > 30).mean()*100:.2f}%",
        fh,
    )

    if auctions_users is not None:
        au = auctions_users[["AUCTION_ID", "PLACEMENT"]].drop_duplicates()
        imps = imps.merge(au, on="AUCTION_ID", how="left")
        for placement, subset in imps[imps["gap_secs"] > 0].groupby("PLACEMENT"):
            series = subset["gap_secs"]
            log(
                f"  Placement {placement}: median={series.median():.2f}s, "
                f"P90={series.quantile(0.9):.2f}s, P99={series.quantile(0.99):.2f}s, "
                f">30s={(series > 30).mean()*100:.2f}%",
                fh,
            )


def placement_transitions(auctions_users, fh):
    log("\n" + "=" * 80, fh)
    log("PLACEMENT TRANSITIONS", fh)
    log("=" * 80, fh)
    if auctions_users is None:
        log("Skipping: auctions_users missing.", fh)
        return

    au = auctions_users.copy()
    au["CREATED_AT"] = pd.to_datetime(au["CREATED_AT"])
    au = au.sort_values(["USER_ID", "CREATED_AT"])
    au["prev_placement"] = au.groupby("USER_ID")["PLACEMENT"].shift(1)
    transitions = au.dropna(subset=["prev_placement"])
    counts = transitions.groupby(["prev_placement", "PLACEMENT"]).size()
    totals = counts.groupby(level=0).sum()
    log("Top transitions:", fh)
    for (prev, curr), cnt in counts.sort_values(ascending=False).head(15).items():
        pct = cnt / totals[prev] * 100
        log(f"  {prev} -> {curr}: {cnt:,} ({pct:.2f}%)", fh)

    starts = au.groupby("PLACEMENT")["prev_placement"].apply(lambda s: s.isna().sum())
    total_starts = starts.sum()
    log("\nSession starts by placement:", fh)
    for placement, cnt in starts.items():
        log(f"  Placement {placement}: {cnt:,} ({cnt/total_starts*100:.2f}%)", fh)


def feature_profiles(session_items, fh):
    log("\n" + "=" * 80, fh)
    log("FEATURE PROFILES", fh)
    log("=" * 80, fh)
    if session_items is None:
        log("Skipping: session_items missing.", fh)
        return

    metrics = (
        session_items.groupby("position")
        .agg(
            mean_bid=("bid", "mean"),
            mean_quality=("quality", "mean"),
            ctr=("clicked", "mean"),
        )
        .reset_index()
    )
    log("First 15 positions:", fh)
    for _, row in metrics[metrics["position"] <= 15].iterrows():
        log(
            f"  Pos {int(row['position']):>2}: bid={row['mean_bid']:.2f}, "
            f"quality={row['mean_quality']:.4f}, CTR={row['ctr']*100:.2f}%",
            fh,
        )

    place_stats = (
        session_items.groupby("placement")
        .agg(
            mean_bid=("bid", "mean"),
            median_bid=("bid", "median"),
            mean_quality=("quality", "mean"),
            median_quality=("quality", "median"),
            ctr=("clicked", "mean"),
        )
        .reset_index()
    )
    log("\nBy placement:", fh)
    for _, row in place_stats.iterrows():
        log(
            f"  Placement {row['placement']}: bid_mean={row['mean_bid']:.2f}, "
            f"bid_median={row['median_bid']:.2f}, "
            f"qual_mean={row['mean_quality']:.4f}, "
            f"qual_median={row['median_quality']:.4f}, "
            f"CTR={row['ctr']*100:.2f}%",
            fh,
        )


def main():
    parser = argparse.ArgumentParser(description="Session & journey EDA script.")
    parser.add_argument("--round", choices=["round1", "round2"], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"12_session_journey_eda_{args.round}.txt"

    with open(output_path, "w") as fh:
        log(f"Session Journey EDA for {args.round}", fh)
        log("=" * 80, fh)

        auctions_users = load_parquet(
            paths.get("auctions_users"),
            columns=["AUCTION_ID", "USER_ID", "PLACEMENT", "CREATED_AT"],
        )
        impressions = load_parquet(
            paths.get("impressions"),
            columns=["AUCTION_ID", "PRODUCT_ID", "USER_ID", "OCCURRED_AT"],
        )
        clicks = load_parquet(
            paths.get("clicks"),
            columns=["AUCTION_ID", "PRODUCT_ID", "USER_ID", "OCCURRED_AT"],
        )
        session_items = load_parquet(paths.get("session_items"))

        if session_items is not None:
            describe_session_items(session_items, fh)
            rank_vs_position(session_items, fh)
            feature_profiles(session_items, fh)
        else:
            log(
                "\nSession-level parquet not available; skipping session-specific sections.",
                fh,
            )

        click_timing(impressions, clicks, fh)
        search_to_impression(auctions_users, impressions, fh)
        pagination_gaps(auctions_users, impressions, fh)
        placement_transitions(auctions_users, fh)

        log("\nEDA complete.", fh)
        log(f"Results written to {output_path}", fh)


if __name__ == "__main__":
    main()
