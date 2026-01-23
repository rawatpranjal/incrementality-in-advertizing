#!/usr/bin/env python3
"""
99_combine_results.py
=====================
Combine all results from the Shopping Episode pipeline into FULL_RESULTS.txt.

Pipeline:
    01_data_pull.py  -> results/01_data_pull.txt
    02_eda.py        -> results/02_eda.txt
    03_modeling.py   -> results/03_modeling.txt

Output:
    results/FULL_RESULTS.txt (concatenation of all)
"""

import os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "FULL_RESULTS.txt"

RESULT_FILES = [
    "01_data_pull.txt",
    "02_eda.txt",
    "03_modeling.txt",
]

SEPARATOR = "\n" + "=" * 80 + "\n"


def main():
    """Combine all result files into FULL_RESULTS.txt."""

    print("=" * 80)
    print("SHOPPING EPISODE INCREMENTALITY ANALYSIS")
    print("COMBINED RESULTS")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Results directory: {RESULTS_DIR.absolute()}")
    print()

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory '{RESULTS_DIR}' does not exist.")
        print("Run the pipeline scripts first: 01_data_pull.py, 02_eda.py, 03_modeling.py")
        return

    combined_content = []
    combined_content.append("=" * 80)
    combined_content.append("SHOPPING EPISODE INCREMENTALITY ANALYSIS")
    combined_content.append("COMBINED RESULTS")
    combined_content.append("=" * 80)
    combined_content.append(f"\nGenerated: {datetime.now().isoformat()}")
    combined_content.append("\nPipeline:")
    combined_content.append("  01_data_pull.py  - Data extraction and episode construction")
    combined_content.append("  02_eda.py        - Exploratory data analysis and validation")
    combined_content.append("  03_modeling.py   - Three analytical models")
    combined_content.append("")

    files_found = []
    files_missing = []

    for filename in RESULT_FILES:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            files_found.append(filename)
        else:
            files_missing.append(filename)

    print("Files found:")
    for f in files_found:
        size = (RESULTS_DIR / f).stat().st_size
        print(f"  [OK] {f} ({size:,} bytes)")

    if files_missing:
        print("\nFiles missing:")
        for f in files_missing:
            print(f"  [MISSING] {f}")

    print()

    combined_content.append("-" * 80)
    combined_content.append("FILES INCLUDED")
    combined_content.append("-" * 80)
    for f in files_found:
        combined_content.append(f"  [OK] {f}")
    for f in files_missing:
        combined_content.append(f"  [MISSING] {f}")
    combined_content.append("")

    for filename in RESULT_FILES:
        filepath = RESULTS_DIR / filename

        combined_content.append(SEPARATOR)
        combined_content.append(f"FILE: {filename}")
        combined_content.append(SEPARATOR)

        if filepath.exists():
            print(f"Reading {filename}...")
            with open(filepath, 'r') as f:
                content = f.read()
            combined_content.append(content)
        else:
            combined_content.append(f"[FILE NOT FOUND: {filename}]")
            combined_content.append("Run the corresponding pipeline script to generate this file.")

        combined_content.append("")

    combined_content.append(SEPARATOR)
    combined_content.append("END OF COMBINED RESULTS")
    combined_content.append(SEPARATOR)

    full_output = "\n".join(combined_content)

    with open(OUTPUT_FILE, 'w') as f:
        f.write(full_output)

    output_size = OUTPUT_FILE.stat().st_size
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total size: {output_size:,} bytes")
    print(f"Files combined: {len(files_found)}/{len(RESULT_FILES)}")

    if files_missing:
        print(f"\nWARNING: {len(files_missing)} file(s) missing. Run all pipeline scripts.")
    else:
        print("\nSUCCESS: All files combined.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
