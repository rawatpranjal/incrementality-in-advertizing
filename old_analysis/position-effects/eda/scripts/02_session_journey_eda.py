#!/usr/bin/env python3
"""
Session and Journey EDA

Consolidated exploratory analysis focused on how placements/rank drive clicks:
    * Placement/session composition
    * Rank vs. on-screen position exposure
    * Session length buckets & click propensity
    * Timeline integrity: search -> impression -> click
    * Pagination/viewability gaps per placement
    * Cross-placement transitions within users
    * Rank-feature summaries (bids, qualities)
    * Device-type proxies (mobile-like vs desktop-like) inferred from viewport size
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

    session_start = (
        imps.groupby("AUCTION_ID")["OCCURRED_AT"].min().rename("session_start")
    )
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
    log("FEATURE PROFILES BY POSITION/PLACEMENT", fh)
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


def viewport_profiles(session_items, fh):
    log("\n" + "=" * 80, fh)
    log("VIEWPORT RECONSTRUCTION", fh)
    log("=" * 80, fh)
    if session_items is None or len(session_items) == 0:
        log("Skipping: session_items missing, cannot compute viewport slots.", fh)
        return

    session_level = (
        session_items[["auction_id", "placement", "n_items"]]
        .drop_duplicates()
        .rename(
            columns={
                "auction_id": "auction_id",
                "placement": "placement",
                "n_items": "viewport_slots",
            }
        )
    )

    buckets = pd.cut(
        session_level["viewport_slots"],
        bins=[0, 2, 4, 7, 12, np.inf],
        labels=["<=2 slots", "3-4 slots", "5-7 slots", "8-12 slots", "13+ slots"],
        right=True,
        include_lowest=True,
    )
    session_level["viewport_bucket"] = buckets

    log("Overall viewport slot distribution:", fh)
    bucket_counts = session_level["viewport_bucket"].value_counts().sort_index()
    total_sessions = len(session_level)
    for bucket, cnt in bucket_counts.items():
        log(f"  {bucket}: {cnt:,} sessions ({cnt/total_sessions*100:.2f}%)", fh)

    log("\nBy placement (median slots and bucket mix):", fh)
    for placement, subset in session_level.groupby("placement"):
        if len(subset) == 0:
            continue
        median_slots = subset["viewport_slots"].median()
        mean_slots = subset["viewport_slots"].mean()
        bucket_mix = subset["viewport_bucket"].value_counts().sort_index()
        log(
            f"  Placement {placement}: median_slots={median_slots:.1f}, "
            f"mean_slots={mean_slots:.1f}",
            fh,
        )
        for bucket, cnt in bucket_mix.items():
            log(f"    {bucket}: {cnt:,} ({cnt/len(subset)*100:.2f}%)", fh)

    log(
        "\nInterpretation: placements dominated by <=2 or 3-4 slots align with narrow/mobile "
        "viewports (e.g., search/category on phones exposing ~2 ads at a time), while >=8 slots "
        "suggest desktop grids or product-page carousels exposing many ads simultaneously.",
        fh,
    )


def product_page_carousel_analysis(session_items, fh):
    log("\n" + "=" * 80, fh)
    log("PRODUCT PAGE CAROUSEL IDENTIFICATION", fh)
    log("=" * 80, fh)
    if session_items is None or len(session_items) == 0:
        log("Skipping: session_items missing.", fh)
        return

    session_level = (
        session_items[["auction_id", "placement", "n_items", "n_clicks"]]
        .drop_duplicates()
        .rename(
            columns={
                "auction_id": "auction_id",
                "placement": "placement",
                "n_items": "viewport_slots",
                "n_clicks": "session_clicks",
            }
        )
    )

    total_sessions = len(session_level)
    p5 = session_level[session_level["placement"] == "5"]
    if len(p5) == 0:
        log("No placement 5 (product page) sessions in dataset.", fh)
        return

    pdp = p5[p5["viewport_slots"] >= 8]
    log(
        f"P5 sessions (product-page context): {len(p5):,} ({len(p5)/total_sessions*100:.2f}% of all sessions)",
        fh,
    )
    log(
        f"  Product page carousels (>=8 slots): {len(pdp):,} "
        f"({len(pdp)/len(p5)*100:.2f}% of P5 sessions)",
        fh,
    )

    if len(pdp) == 0:
        log("No P5 sessions meet the >=8 slot carousel heuristic.", fh)
        return

    slot_counts = pdp["viewport_slots"].value_counts().sort_index()
    log("\nTop carousel slot counts:", fh)
    for slots, cnt in slot_counts.head(10).items():
        log(f"  {int(slots)} slots: {cnt:,} sessions ({cnt/len(pdp)*100:.2f}%)", fh)

    log(
        f"\nCarousel session stats: median_slots={pdp['viewport_slots'].median():.1f}, "
        f"mean_slots={pdp['viewport_slots'].mean():.1f}, "
        f"median_clicks={pdp['session_clicks'].median():.2f}, "
        f"%with_click={(pdp['session_clicks']>0).mean()*100:.2f}%",
        fh,
    )

    first_slots = pdp["viewport_slots"].iloc[:10].tolist()
    log(f"\nExample viewport sizes (first 10): {first_slots}", fh)


def device_proxy_checks(auctions_users, impressions, clicks, session_items, fh):
    log("\n" + "=" * 80, fh)
    log("DEVICE PROXY (MOBILE-LIKE VS DESKTOP-LIKE)", fh)
    log("=" * 80, fh)
    if impressions is None:
        log("Skipping: impressions missing, cannot infer viewport size.", fh)
        return

    imp_counts = (
        impressions.groupby("AUCTION_ID")
        .size()
        .rename("impression_count")
        .reset_index()
    )
    if len(imp_counts) == 0:
        log("No impression rows available.", fh)
        return

    session_df = imp_counts
    if clicks is not None and len(clicks) > 0:
        click_counts = (
            clicks.groupby("AUCTION_ID").size().rename("click_count").reset_index()
        )
        session_df = session_df.merge(click_counts, on="AUCTION_ID", how="left")
    else:
        session_df = session_df.assign(click_count=0)
    session_df["click_count"] = session_df["click_count"].fillna(0).astype(int)

    # Attach viewport slot estimates and placements when available.
    if session_items is not None and len(session_items) > 0:
        viewport_info = (
            session_items[["auction_id", "n_items", "placement"]]
            .drop_duplicates()
            .rename(
                columns={
                    "auction_id": "AUCTION_ID",
                    "n_items": "viewport_slots",
                    "placement": "SESSION_PLACEMENT",
                }
            )
        )
        session_df = session_df.merge(viewport_info, on="AUCTION_ID", how="left")
    else:
        session_df["viewport_slots"] = np.nan
        session_df["SESSION_PLACEMENT"] = np.nan

    if auctions_users is not None:
        placement_fallback = (
            auctions_users[["AUCTION_ID", "PLACEMENT"]]
            .drop_duplicates()
            .rename(columns={"PLACEMENT": "FALLBACK_PLACEMENT"})
        )
        session_df = session_df.merge(placement_fallback, on="AUCTION_ID", how="left")
        if "SESSION_PLACEMENT" in session_df.columns:
            session_df["SESSION_PLACEMENT"] = session_df["SESSION_PLACEMENT"].fillna(
                session_df["FALLBACK_PLACEMENT"]
            )
        else:
            session_df["SESSION_PLACEMENT"] = session_df["FALLBACK_PLACEMENT"]
        session_df = session_df.drop(columns=["FALLBACK_PLACEMENT"])

    if "SESSION_PLACEMENT" not in session_df.columns:
        session_df["SESSION_PLACEMENT"] = np.nan
    if "viewport_slots" not in session_df.columns:
        session_df["viewport_slots"] = np.nan

    def classify(count):
        if pd.isna(count):
            return "unknown"
        if count <= 4:
            return "mobile-like (<=4 slots)"
        if count >= 8:
            return "desktop-like (>=8 slots)"
        return "mixed (5-7 slots)"

    session_df["device_guess"] = session_df["impression_count"].apply(classify)
    session_df["device_source"] = "impression_count"
    if session_items is not None and len(session_items) > 0:
        mask = session_df["viewport_slots"].notna()
        session_df.loc[mask, "device_guess"] = session_df.loc[mask, "viewport_slots"].apply(
            classify
        )
        session_df.loc[mask, "device_source"] = "viewport_slots"

    total_sessions = len(session_df)
    log(
        "Heuristic: sessions rendering ≤4 products resemble narrow/mobile viewports, "
        "≥8 products resemble wide/desktop layouts; 5-7 treated as mixed exposure.",
        fh,
    )
    log(
        "Known layout references: mobile search/category viewports usually surface ~2 ads "
        "at a time, desktop search/category tiles show ~4 ads simultaneously, "
        "and product detail pages can surface ~8 \"Sponsored Listings\" in-carousels plus "
        "bottom \"More like this\" modules.",
        fh,
    )

    source_counts = session_df["device_source"].value_counts()
    log("\nDevice classification data sources:", fh)
    for source, cnt in source_counts.items():
        log(f"  {source}: {cnt:,} sessions ({cnt/total_sessions*100:.2f}%)", fh)

    for bucket, group in session_df.groupby("device_guess"):
        sessions = len(group)
        share = sessions / total_sessions * 100
        total_clicks = group["click_count"].sum()
        total_imps = group["impression_count"].sum()
        ctr = (total_clicks / total_imps * 100) if total_imps > 0 else 0
        pct_with_click = (group["click_count"] > 0).mean() * 100
        log(
            f"  {bucket}: sessions={sessions:,} ({share:.2f}%), "
            f"median_imps={group['impression_count'].median():.1f}, "
            f"mean_imps={group['impression_count'].mean():.1f}, "
            f"%sessions_with_click={pct_with_click:.2f}%, "
            f"mean_clicks={group['click_count'].mean():.3f}, "
            f"CTR={ctr:.2f}%",
            fh,
        )

    log("\nPlacement mix within each device proxy:", fh)
    for bucket, subset in session_df.groupby("device_guess"):
        counts = subset["SESSION_PLACEMENT"].value_counts(dropna=False)
        total = counts.sum()
        log(f"  {bucket}:", fh)
        for placement_value, cnt in counts.items():
            label = "Unknown" if pd.isna(placement_value) else placement_value
            log(
                f"    Placement {label}: {cnt:,} ({cnt/total*100:.2f}%)",
                fh,
            )


def main():
    parser = argparse.ArgumentParser(description="Session & journey EDA script.")
    parser.add_argument("--round", choices=["round1", "round2"], required=True)
    args = parser.parse_args()

    paths = get_paths(args.round)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"02_session_journey_eda_{args.round}.txt"

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
            viewport_profiles(session_items, fh)
            product_page_carousel_analysis(session_items, fh)
        else:
            log(
                "\nSession-level parquet not available for this round; "
                "skipping session-specific sections.",
                fh,
            )

        click_timing(impressions, clicks, fh)
        search_to_impression(auctions_users, impressions, fh)
        pagination_gaps(auctions_users, impressions, fh)
        placement_transitions(auctions_users, fh)
        device_proxy_checks(auctions_users, impressions, clicks, session_items, fh)

        log("\nEDA complete.", fh)
        log(f"Results written to {output_path}", fh)


if __name__ == "__main__":
    main()
