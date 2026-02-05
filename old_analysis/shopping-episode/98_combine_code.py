#!/usr/bin/env python3
"""
98_combine_code.py
Combines all .py files into a single FULL_CODE.txt for AI learning.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "FULL_CODE.txt"

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Find all .py files, sorted by name
    py_files = sorted(BASE_DIR.glob("*.py"))

    with open(OUTPUT_FILE, 'w') as out:
        out.write("=" * 80 + "\n")
        out.write("FULL CODE REPOSITORY\n")
        out.write("=" * 80 + "\n\n")
        out.write(f"Total scripts: {len(py_files)}\n")
        out.write("Files: " + ", ".join(f.name for f in py_files) + "\n\n")

        for py_file in py_files:
            out.write("=" * 80 + "\n")
            out.write(f"FILE: {py_file.name}\n")
            out.write("=" * 80 + "\n\n")

            try:
                content = py_file.read_text()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as e:
                out.write(f"ERROR reading file: {e}\n")

            out.write("\n")

    print(f"Combined {len(py_files)} files into {OUTPUT_FILE}")
    print(f"Files: {[f.name for f in py_files]}")

if __name__ == "__main__":
    main()
