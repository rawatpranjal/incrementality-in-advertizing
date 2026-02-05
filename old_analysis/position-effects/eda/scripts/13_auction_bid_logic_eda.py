#!/usr/bin/env python3
"""
Auction Bid Logic EDA

Sanity-checks the claimed relationships:
    * Retrieval breadth (bidders per auction) and ranking by quality*bid
    * Bid ~= CVR * AOV / target_ROAS (implied target distribution)
    * Pacing multiplier adjusts bids (mostly 1, <1 when throttled)
    * Quality calibration: higher QUALITY => higher observed CTR (impressions/clicks)
    * CVR calibration (round2): higher CONVERSION_RATE => higher observed post-click conv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from textwrap import dedent

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
            "auctions_results": DATA_DIR / "round1/auctions_results_all.parquet",
            "impressions": DATA_DIR / "round1/impressions_all.parquet",
            "clicks": DATA_DIR / "round1/clicks_all.parquet",
            "catalog": DATA_DIR / "round1/catalog_all.parquet",
        }
    if round_name == "round2":
        return {
            "auctions_results": DATA_DIR / "round2/auctions_results_r2.parquet",
            "impressions": DATA_DIR / "round2/impressions_r2.parquet",
            "clicks": DATA_DIR / "round2/clicks_r2.parquet",
            "catalog": DATA_DIR / "round2/catalog_r2.parquet",
            "purchases": DATA_DIR / "round2/purchases_r2.parquet",
        }
    raise ValueError(f"Unknown round {round_name}")

def analyze(round_name: str, fh):
    paths = get_paths(round_name)
    log("=" * 80, fh)
    log(f"AUCTION BID LOGIC EDA ({round_name})", fh)
    log("=" * 80, fh)
    log(f"Sources: {paths}", fh)

    cols = [
        "RANKING",
        "IS_WINNER",
        "FINAL_BID",
        "QUALITY",
        "CONVERSION_RATE",
        "PACING",
        "PRICE",
    ]
    df = pd.read_parquet(paths["auctions_results"], columns=cols)
    log(f"Loaded {len(df):,} auction rows", fh)

    # 1. Ranking vs quality * bid
    df["bid_quality_score"] = df["FINAL_BID"] * df["QUALITY"]
    corr_rank_bq = df["RANKING"].corr(-df["bid_quality_score"])
    corr_rank_bid = df["RANKING"].corr(-df["FINAL_BID"])
    corr_rank_quality = df["RANKING"].corr(-df["QUALITY"])
    log("\nRanking correlations (negative sign = better rank with higher value):", fh)
    log(f"  Corr(rank, bid*quality)   : {corr_rank_bq:.4f}", fh)
    log(f"  Corr(rank, bid)           : {corr_rank_bid:.4f}", fh)
    log(f"  Corr(rank, quality)       : {corr_rank_quality:.4f}", fh)

    df["score_rank_percentile"] = (
        pd.Series(df["bid_quality_score"]).rank(pct=True)
    )
    rank_summary = df[["RANKING", "bid_quality_score"]].describe(percentiles=[0.1,0.5,0.9])
    log("\nSummary stats (RANKING, bid*quality score):", fh)
    log(str(rank_summary), fh)

    # Retrieval breadth: bidders per auction
    log("\nRetrieval breadth (bidders per auction):", fh)
    bidders = pd.read_parquet(paths["auctions_results"], columns=["AUCTION_ID"]) \
        .groupby("AUCTION_ID").size().rename("bidders")
    log(str(bidders.describe(percentiles=[0.1,0.5,0.9,0.99])), fh)

    # 2. Bid vs CVR * price / target
    ratio_mask = (df["CONVERSION_RATE"] > 0) & (df["PRICE"].notna())
    ratio_df = df[ratio_mask].copy()
    ratio_df["cvr_price"] = ratio_df["CONVERSION_RATE"] * ratio_df["PRICE"]
    ratio_df["bid_over_cvr_price"] = ratio_df["FINAL_BID"] / ratio_df["cvr_price"]
    log(
        "\nBid vs CVR*PRICE summary (only rows with price & CVR>0):", fh
    )
    log(f"  Rows used: {len(ratio_df):,}", fh)
    log(
        str(
            ratio_df[["FINAL_BID", "cvr_price", "bid_over_cvr_price"]].describe(
                percentiles=[0.1, 0.5, 0.9]
            )
        ),
        fh,
    )
    log(
        dedent(
            """
            Interpretation: final_bid ≈ cvr * price / target_roas.
            If target ROAS ~ 10x, bid_over_cvr_price should cluster near 0.1.
            """
        ).strip(),
        fh,
    )

    # 3. Pacing multiplier inspection
    pacing_desc = df["PACING"].describe(percentiles=[0.5, 0.9, 0.99])
    pacing_lt_one = (df["PACING"] < 1).mean() * 100
    pacing_hist = df["PACING"].value_counts(bins=[0,0.5,0.8,0.9,0.95,1]).sort_index()
    log("\nPacing multiplier stats:", fh)
    log(str(pacing_desc), fh)
    log(f"Rows with pacing < 1: {pacing_lt_one:.2f}%", fh)
    log("Pacing histogram (value ranges -> counts):", fh)
    for interval, count in pacing_hist.items():
        log(f"  {interval}: {count:,}", fh)

    # Winner ratio vs score
    high_score = df["bid_quality_score"].quantile(0.9)
    low_score = df["bid_quality_score"].quantile(0.1)
    win_rate_high = df[df["bid_quality_score"] >= high_score]["IS_WINNER"].mean()
    win_rate_low = df[df["bid_quality_score"] <= low_score]["IS_WINNER"].mean()
    log("\nWin rates by bid*quality percentile:", fh)
    log(f"  Top 10% score win rate: {win_rate_high*100:.2f}%", fh)
    log(f"  Bottom 10% score win rate: {win_rate_low*100:.2f}%", fh)

    # 4. Quality calibration: observed CTR by QUALITY decile
    try:
        imps = pd.read_parquet(paths["impressions"], columns=["AUCTION_ID","PRODUCT_ID"])
        clks = pd.read_parquet(paths["clicks"], columns=["AUCTION_ID","PRODUCT_ID"])
        # Aggregate to counts per (AUCTION_ID, PRODUCT_ID)
        imp_counts = imps.groupby(["AUCTION_ID","PRODUCT_ID"]).size().rename("imps").reset_index()
        clk_counts = clks.groupby(["AUCTION_ID","PRODUCT_ID"]).size().rename("clks").reset_index()
        joined = imp_counts.merge(clk_counts, on=["AUCTION_ID","PRODUCT_ID"], how="left")
        joined["clks"] = joined["clks"].fillna(0)
        # Bring QUALITY
        quality_map = pd.read_parquet(paths["auctions_results"], columns=["AUCTION_ID","PRODUCT_ID","QUALITY"]).drop_duplicates()
        joined = joined.merge(quality_map, on=["AUCTION_ID","PRODUCT_ID"], how="left")
        joined = joined.dropna(subset=["QUALITY"])  # need quality
        joined["quality_decile"] = pd.qcut(joined["QUALITY"], 10, labels=False, duplicates='drop')
        ctr_by_dec = joined.groupby("quality_decile").apply(lambda g: g["clks"].sum()/g["imps"].sum()).rename("ctr").reset_index()
        log("\nObserved CTR by QUALITY decile (0=lowest):", fh)
        for _, row in ctr_by_dec.iterrows():
            log(f"  decile {int(row['quality_decile'])}: CTR={row['ctr']*100:.2f}%", fh)
    except Exception as e:
        log(f"\nQuality calibration skipped: {e}", fh)

    # 5. CVR calibration (round2 only): observed post-click conv by CONVERSION_RATE decile
    if "purchases" in paths and paths["purchases"].exists():
        try:
            clks = pd.read_parquet(paths["clicks"], columns=["AUCTION_ID","PRODUCT_ID","USER_ID"])  # clicks are per user
            conv = pd.read_parquet(paths["purchases"], columns=["PRODUCT_ID","USER_ID"])  # assume CVR tied to user+product
            # Click-level conversion flag
            purchase_keys = set(zip(conv["USER_ID"], conv["PRODUCT_ID"]))
            clks["purchased"] = clks.apply(lambda r: (r["USER_ID"], r["PRODUCT_ID"]) in purchase_keys, axis=1)
            # Attach model CONVERSION_RATE from auctions_results at (AUCTION_ID, PRODUCT_ID)
            conv_map = pd.read_parquet(paths["auctions_results"], columns=["AUCTION_ID","PRODUCT_ID","CONVERSION_RATE"]).drop_duplicates()
            clks = clks.merge(conv_map, on=["AUCTION_ID","PRODUCT_ID"], how="left")
            clks = clks.dropna(subset=["CONVERSION_RATE"])
            clks["cvr_decile"] = pd.qcut(clks["CONVERSION_RATE"], 10, labels=False, duplicates='drop')
            lift = clks.groupby("cvr_decile")["purchased"].mean().rename("post_click_cvr").reset_index()
            log("\nObserved post-click CVR by CONVERSION_RATE decile (round2):", fh)
            for _, row in lift.iterrows():
                log(f"  decile {int(row['cvr_decile'])}: post-click CVR={row['post_click_cvr']*100:.2f}%", fh)
        except Exception as e:
            log(f"\nCVR calibration skipped: {e}", fh)

def main():
    parser = argparse.ArgumentParser(description="Auction bid logic EDA")
    parser.add_argument("--round", choices=["round1","round2"], required=True)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"13_auction_bid_logic_eda_{args.round}.txt"
    with open(out, "w") as fh:
        analyze(args.round, fh)
        log(f"\nEDA complete. Results: {out}", fh)

if __name__ == "__main__":
    main()
