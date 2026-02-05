#!/usr/bin/env bash
set -euo pipefail

# Fetch canonical PDFs (where openly available) for key references into paper/sources.
# Some articles are paywalled; for those we store DOI links in refs_links.txt instead.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/sources"
mkdir -p "$OUT_DIR"

echo "Saving references into: $OUT_DIR"

# Cohen et al. (2016) NBER Working Paper 22627 (open PDF)
COHEN_URL="https://www.nber.org/system/files/working_papers/w22627/w22627.pdf"
echo "Downloading Cohen et al. (2016) from NBER..."
curl -L "$COHEN_URL" -o "$OUT_DIR/cohen_etal_2016_uber_nber22627.pdf"

# Lee & Lemieux (2010), JEL. PDF availability may require access; keep DOI link in refs_links.txt
echo "Lee & Lemieux (2010) likely paywalled; keeping DOI link in refs_links.txt"

# Varian (2007) Position auctions (IJIO) is typically paywalled; keep DOI link in refs_links.txt
echo "Varian (2007) likely paywalled; keeping DOI link in refs_links.txt"

echo "Done. For restricted articles, please download via institutional access and place PDFs into $OUT_DIR."

