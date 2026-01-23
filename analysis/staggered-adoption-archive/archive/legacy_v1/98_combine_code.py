#!/usr/bin/env python3
"""
Staggered Adoption DiD Analysis: Combine Code
Concatenates all *.py files into FULL_CODE.txt for AI reasoning.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "FULL_CODE.txt"

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get all .py files in order
    py_files = sorted([
        f for f in BASE_DIR.glob("*.py")
    ])

    print(f"Found {len(py_files)} Python files to combine")

    with open(OUTPUT_FILE, 'w') as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write("STAGGERED ADOPTION DiD ANALYSIS - FULL CODE\n")
        outfile.write("=" * 80 + "\n")
        outfile.write("\n")
        outfile.write(f"Combined from {len(py_files)} Python files\n")
        outfile.write("\n")

        for py_file in py_files:
            outfile.write("\n")
            outfile.write("#" * 80 + "\n")
            outfile.write(f"# FILE: {py_file.name}\n")
            outfile.write("#" * 80 + "\n")
            outfile.write("\n")

            try:
                with open(py_file, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"# ERROR reading file: {str(e)}\n")

            print(f"  Added: {py_file.name}")

    print(f"\nCombined code saved to: {OUTPUT_FILE}")
    print(f"Total size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
