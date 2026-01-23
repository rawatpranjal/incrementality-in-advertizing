#!/usr/bin/env python3
"""
Combine all EDA results into tabulated summary.
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EDA_DIR = Path(__file__).parent
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "99_EDA_SUMMARY.txt"

def extract_key_metrics():
    """Extract key metrics from each EDA result file."""
    metrics = {}

    # Q2: Vendor Panel Balance
    f02 = RESULTS_DIR / "02_vendor_panel_balance.txt"
    if f02.exists():
        text = f02.read_text()
        m = {}
        match = re.search(r"Unique bidding vendors: ([\d,]+)", text)
        if match: m["bidding_vendors"] = match.group(1)
        match = re.search(r"Unique catalog vendors: ([\d,]+)", text)
        if match: m["catalog_vendors"] = match.group(1)
        match = re.search(r"Bidder coverage in catalog: ([\d.]+%)", text)
        if match: m["bidder_coverage"] = match.group(1)
        metrics["Q02_panel_balance"] = m

    # Q3: Treatment Absorbing
    f03 = RESULTS_DIR / "03_treatment_absorbing.txt"
    if f03.exists():
        text = f03.read_text()
        m = {}
        match = re.search(r"P\(Turn OFF \| was ON\):\s+([\d.]+%)", text)
        if match: m["flicker_rate"] = match.group(1)
        match = re.search(r"P\(Stay ON \| was ON\):\s+([\d.]+%)", text)
        if match: m["persistence_rate"] = match.group(1)
        match = re.search(r"Vendors who NEVER flicker.*?: ([\d,]+) \(([\d.]+%)\)", text)
        if match: m["never_flicker_pct"] = match.group(2)
        match = re.search(r"\[(\w+)\].*absorbing", text, re.IGNORECASE)
        if match: m["status"] = match.group(1)
        else: m["status"] = "OK" if float(m.get("flicker_rate", "100%").replace("%","")) < 20 else "WARNING"
        metrics["Q03_absorbing"] = m

    # Q4: Adoption Velocity
    f04 = RESULTS_DIR / "04_adoption_velocity.txt"
    if f04.exists():
        text = f04.read_text()
        m = {}
        match = re.search(r"Number of cohort weeks: (\d+)", text)
        if match: m["n_cohorts"] = match.group(1)
        match = re.search(r"% adopted in Week 1: ([\d.]+%)", text)
        if match: m["week1_pct"] = match.group(1)
        match = re.search(r"% adopted by Week 3: ([\d.]+%)", text)
        if match: m["week3_pct"] = match.group(1)
        match = re.search(r"Largest cohort: ([\d.]+%) of treated", text)
        if match: m["largest_cohort_pct"] = match.group(1)
        match = re.search(r"\[(\w+)\]", text)
        if match: m["status"] = match.group(1)
        else: m["status"] = "OK" if float(m.get("week1_pct", "100%").replace("%","")) < 50 else "WARNING"
        metrics["Q04_adoption"] = m

    # Q5: Ashenfelter's Dip
    f05 = RESULTS_DIR / "05_ashenfelter_dip.txt"
    if f05.exists():
        text = f05.read_text()
        m = {}
        match = re.search(r"Pre-treatment observations \(relative_week < 0\): ([\d,]+)", text)
        if match: m["pre_obs"] = match.group(1)
        match = re.search(r"Treated vendors in sample: ([\d,]+)", text)
        if match: m["treated_vendors"] = match.group(1)
        match = re.search(r"\[(\w+)\]", text)
        if match: m["status"] = match.group(1)
        metrics["Q05_ashenfelter"] = m

    # Q8: Zero-Inflation
    f08 = RESULTS_DIR / "08_zero_inflation.txt"
    if f08.exists():
        text = f08.read_text()
        m = {}
        for var in ["promoted_gmv", "impressions", "clicks", "wins", "total_spend"]:
            pattern = rf"{var}\s+[\d,]+\s+[\d,]+\s+([\d.]+%)"
            match = re.search(pattern, text)
            if match: m[f"{var}_zero_pct"] = match.group(1)
        match = re.search(r"\[(CRITICAL|WARNING|OK)\].*zero", text, re.IGNORECASE)
        if match: m["status"] = match.group(1)
        metrics["Q08_zero_inflation"] = m

    # Q9: Whale Concentration
    f09 = RESULTS_DIR / "09_whale_concentration.txt"
    if f09.exists():
        text = f09.read_text()
        m = {}
        match = re.search(r"Top 1%: ([\d.]+%) of GMV", text)
        if match: m["top1_gmv"] = match.group(1)
        match = re.search(r"Top 10%: ([\d.]+%) of GMV", text)
        if match: m["top10_gmv"] = match.group(1)
        match = re.search(r"GMV Gini coefficient: ([\d.]+)", text)
        if match: m["gini_gmv"] = match.group(1)
        match = re.search(r"\[(\w+)\].*concentration", text, re.IGNORECASE)
        if match: m["status"] = match.group(1)
        metrics["Q09_whale"] = m

    # Q10: Organic Cannibalization
    f10 = RESULTS_DIR / "10_organic_cannibalization.txt"
    if f10.exists():
        text = f10.read_text()
        m = {}
        match = re.search(r"Post-treatment observations: ([\d,]+)", text)
        if match: m["post_obs"] = match.group(1)
        match = re.search(r"Correlation\(impressions, conversion_rate\): ([-\d.]+)", text)
        if match: m["corr_imp_conv"] = match.group(1)
        match = re.search(r"Efficiency ratio \(Q5/Q1\): ([\d.]+)", text)
        if match: m["efficiency_ratio"] = match.group(1)
        metrics["Q10_cannibalization"] = m

    return metrics

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("EDA VALIDATION SUMMARY - STAGGERED ADOPTION DiD\n")
        f.write("=" * 100 + "\n\n")

        metrics = extract_key_metrics()

        # Table 1: Data Quality
        f.write("-" * 100 + "\n")
        f.write("TABLE 1: DATA QUALITY CHECKS\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"{'Question':<40} {'Metric':<30} {'Value':<20} {'Status':<10}\n")
        f.write(f"{'-'*40} {'-'*30} {'-'*20} {'-'*10}\n")

        if "Q02_panel_balance" in metrics:
            m = metrics["Q02_panel_balance"]
            f.write(f"{'Q2: Vendor Panel Balance':<40} {'Bidding Vendors':<30} {m.get('bidding_vendors', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Catalog Vendors':<30} {m.get('catalog_vendors', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Bidder Coverage':<30} {m.get('bidder_coverage', 'N/A'):<20} {'CHECK':<10}\n")

        if "Q08_zero_inflation" in metrics:
            m = metrics["Q08_zero_inflation"]
            f.write(f"{'Q8: Zero-Inflation':<40} {'promoted_gmv Zeros':<30} {m.get('promoted_gmv_zero_pct', 'N/A'):<20} {m.get('status', ''):<10}\n")
            f.write(f"{'':<40} {'impressions Zeros':<30} {m.get('impressions_zero_pct', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'clicks Zeros':<30} {m.get('clicks_zero_pct', 'N/A'):<20} {'':<10}\n")

        f.write("\n")

        # Table 2: Identification Assumptions
        f.write("-" * 100 + "\n")
        f.write("TABLE 2: IDENTIFICATION ASSUMPTIONS\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"{'Question':<40} {'Metric':<30} {'Value':<20} {'Status':<10}\n")
        f.write(f"{'-'*40} {'-'*30} {'-'*20} {'-'*10}\n")

        if "Q03_absorbing" in metrics:
            m = metrics["Q03_absorbing"]
            f.write(f"{'Q3: Treatment Absorbing':<40} {'Flicker Rate (ON→OFF)':<30} {m.get('flicker_rate', 'N/A'):<20} {m.get('status', ''):<10}\n")
            f.write(f"{'':<40} {'Persistence Rate':<30} {m.get('persistence_rate', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Never Flicker %':<30} {m.get('never_flicker_pct', 'N/A'):<20} {'':<10}\n")

        if "Q04_adoption" in metrics:
            m = metrics["Q04_adoption"]
            f.write(f"{'Q4: Adoption Velocity':<40} {'N Cohorts':<30} {m.get('n_cohorts', 'N/A'):<20} {m.get('status', ''):<10}\n")
            f.write(f"{'':<40} {'Week 1 Adoption':<30} {m.get('week1_pct', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Largest Cohort %':<30} {m.get('largest_cohort_pct', 'N/A'):<20} {'':<10}\n")

        if "Q05_ashenfelter" in metrics:
            m = metrics["Q05_ashenfelter"]
            f.write(f"{'Q5: Ashenfelter Dip':<40} {'Pre-treatment Obs':<30} {m.get('pre_obs', 'N/A'):<20} {m.get('status', ''):<10}\n")
            f.write(f"{'':<40} {'Treated Vendors':<30} {m.get('treated_vendors', 'N/A'):<20} {'':<10}\n")

        f.write("\n")

        # Table 3: Concentration & Externalities
        f.write("-" * 100 + "\n")
        f.write("TABLE 3: CONCENTRATION & EXTERNALITIES\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"{'Question':<40} {'Metric':<30} {'Value':<20} {'Status':<10}\n")
        f.write(f"{'-'*40} {'-'*30} {'-'*20} {'-'*10}\n")

        if "Q09_whale" in metrics:
            m = metrics["Q09_whale"]
            f.write(f"{'Q9: Whale Concentration':<40} {'Top 1% GMV Share':<30} {m.get('top1_gmv', 'N/A'):<20} {m.get('status', ''):<10}\n")
            f.write(f"{'':<40} {'Top 10% GMV Share':<30} {m.get('top10_gmv', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Gini Coefficient':<30} {m.get('gini_gmv', 'N/A'):<20} {'':<10}\n")

        if "Q10_cannibalization" in metrics:
            m = metrics["Q10_cannibalization"]
            f.write(f"{'Q10: Cannibalization Risk':<40} {'Post-treatment Obs':<30} {m.get('post_obs', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Corr(Imp, ConvRate)':<30} {m.get('corr_imp_conv', 'N/A'):<20} {'':<10}\n")
            f.write(f"{'':<40} {'Efficiency Ratio Q5/Q1':<30} {m.get('efficiency_ratio', 'N/A'):<20} {'':<10}\n")

        f.write("\n")

        # Table 4: Scripts Requiring Snowflake
        f.write("-" * 100 + "\n")
        f.write("TABLE 4: PENDING (REQUIRE SNOWFLAKE)\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"{'Script':<40} {'Purpose':<50} {'Status':<10}\n")
        f.write(f"{'-'*40} {'-'*50} {'-'*10}\n")
        f.write(f"{'01_orphan_rate_gmv.py':<40} {'GMV orphan rate from PURCHASES→CATALOG':<50} {'PENDING':<10}\n")
        f.write(f"{'06_bid_cpc_verification.py':<40} {'FINAL_BID = CPC verification':<50} {'PENDING':<10}\n")
        f.write(f"{'07_auction_rank_determinism.py':<40} {'RANKING ~ FINAL_BID + QUALITY + PACING':<50} {'PENDING':<10}\n")

        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 100 + "\n")

        print(f"Summary written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
