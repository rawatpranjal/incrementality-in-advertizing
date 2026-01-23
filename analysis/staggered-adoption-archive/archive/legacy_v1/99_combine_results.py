#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Combine Results
Concatenates all results/*.txt files into FULL_RESULTS.txt.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "FULL_RESULTS.txt"

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get all .txt files except FULL_RESULTS.txt and FULL_CODE.txt
    txt_files = sorted([
        f for f in RESULTS_DIR.glob("*.txt")
        if f.name not in ("FULL_RESULTS.txt", "FULL_CODE.txt")
    ])

    print(f"Found {len(txt_files)} result files to combine")

    with open(OUTPUT_FILE, 'w') as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write("STAGGERED ADOPTION DiD ANALYSIS - FULL RESULTS\n")
        outfile.write("=" * 80 + "\n")
        outfile.write("\n")
        outfile.write(f"Combined from {len(txt_files)} result files\n")
        outfile.write("\n")

        for txt_file in txt_files:
            outfile.write("\n")
            outfile.write("#" * 80 + "\n")
            outfile.write(f"# FILE: {txt_file.name}\n")
            outfile.write("#" * 80 + "\n")
            outfile.write("\n")

            try:
                with open(txt_file, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"ERROR reading file: {str(e)}\n")

            print(f"  Added: {txt_file.name}")

    print(f"\nCombined results saved to: {OUTPUT_FILE}")
    print(f"Total size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
