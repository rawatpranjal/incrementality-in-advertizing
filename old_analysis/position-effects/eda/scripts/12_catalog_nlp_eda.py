#!/usr/bin/env python3
"""
12_catalog_nlp_eda.py
Text mining and NLP analysis of catalog data.

Analyzes NAME, DESCRIPTION, and CATEGORIES fields to extract:
- Text coverage and quality metrics
- Brand, color, size, category distributions
- TF-IDF vocabulary analysis
- Price correlations with text features

Usage:
    python 12_catalog_nlp_eda.py --round round1
    python 12_catalog_nlp_eda.py --round round2
"""

import argparse
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_BASE = BASE_DIR / "0_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_data_paths(round_name):
    """Return data paths for specified round."""
    if round_name == "round1":
        return {
            'catalog': DATA_BASE / "round1/catalog_all.parquet",
        }
    elif round_name == "round2":
        return {
            'catalog': DATA_BASE / "round2/catalog_r2.parquet",
        }
    else:
        raise ValueError(f"Unknown round: {round_name}")


# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def compute_gini(values):
    """Compute Gini coefficient for concentration measurement."""
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    values = np.sort(values)
    n = len(values)
    if n == 1:
        return 0.0
    cumsum = np.cumsum(values)
    if cumsum[-1] == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def parse_categories(cat_str):
    """
    Parse CATEGORIES JSON array into dict of extracted fields.

    Returns dict with keys: brand, colors, sizes, department_id, category_ids, domain
    """
    result = {
        'brand': None,
        'colors': [],
        'sizes': [],
        'department_id': None,
        'category_ids': [],
        'domain': None
    }

    if pd.isna(cat_str) or cat_str == '':
        return result

    try:
        categories = json.loads(cat_str)
    except (json.JSONDecodeError, TypeError):
        return result

    for item in categories:
        if not isinstance(item, str):
            continue

        if item.startswith('brand#'):
            result['brand'] = item[6:]  # Remove 'brand#' prefix
        elif item.startswith('color#'):
            result['colors'].append(item[6:])
        elif item.startswith('size#'):
            result['sizes'].append(item[5:])
        elif item.startswith('department#'):
            result['department_id'] = item[11:]
        elif item.startswith('category#'):
            result['category_ids'].append(item[9:])
        elif item.startswith('domain#'):
            result['domain'] = item[7:]

    return result


def percentile_stats(series, name, f):
    """Log percentile statistics for a numeric series."""
    valid = series.dropna()
    log(f"  {name}:", f)
    log(f"    count: {len(valid):,}", f)
    log(f"    mean: {valid.mean():.2f}", f)
    log(f"    std: {valid.std():.2f}", f)
    log(f"    min: {valid.min():.0f}", f)
    log(f"    p1: {valid.quantile(0.01):.0f}", f)
    log(f"    p5: {valid.quantile(0.05):.0f}", f)
    log(f"    p25: {valid.quantile(0.25):.0f}", f)
    log(f"    median: {valid.median():.0f}", f)
    log(f"    p75: {valid.quantile(0.75):.0f}", f)
    log(f"    p95: {valid.quantile(0.95):.0f}", f)
    log(f"    p99: {valid.quantile(0.99):.0f}", f)
    log(f"    max: {valid.max():.0f}", f)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Catalog NLP EDA")
    parser.add_argument('--round', type=str, required=True, choices=['round1', 'round2'],
                        help='Data round to analyze')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    paths = get_data_paths(args.round)
    output_file = RESULTS_DIR / f"12_catalog_nlp_eda_{args.round}.txt"

    with open(output_file, 'w') as f:
        log("=" * 80, f)
        log(f"CATALOG NLP EDA: TEXT MINING ANALYSIS ({args.round.upper()})", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Data Loading
        # -----------------------------------------------------------------
        log("Loading catalog data...", f)
        catalog_path = paths['catalog']
        log(f"Path: {catalog_path}", f)
        df = pd.read_parquet(catalog_path)
        log(f"Loaded {len(df):,} products", f)
        log(f"Columns: {df.columns.tolist()}", f)
        log("", f)

        # Check for CATALOG_PRICE vs PRICE column
        price_col = 'CATALOG_PRICE' if 'CATALOG_PRICE' in df.columns else 'PRICE'
        log(f"Using price column: {price_col}", f)
        log("", f)

        # =================================================================
        # SECTION 1: TEXT FIELD COVERAGE
        # =================================================================
        log("=" * 80, f)
        log("SECTION 1: TEXT FIELD COVERAGE", f)
        log("-" * 40, f)
        log("", f)

        total = len(df)

        # NAME field
        name_null = df['NAME'].isna().sum()
        name_empty = (df['NAME'].fillna('') == '').sum() - name_null
        name_valid = total - name_null - name_empty
        log(f"NAME field:", f)
        log(f"  null: {name_null:,} ({100*name_null/total:.2f}%)", f)
        log(f"  empty string: {name_empty:,} ({100*name_empty/total:.2f}%)", f)
        log(f"  valid: {name_valid:,} ({100*name_valid/total:.2f}%)", f)
        log("", f)

        # DESCRIPTION field
        desc_null = df['DESCRIPTION'].isna().sum()
        desc_empty = (df['DESCRIPTION'].fillna('') == '').sum() - desc_null
        desc_valid = total - desc_null - desc_empty
        log(f"DESCRIPTION field:", f)
        log(f"  null: {desc_null:,} ({100*desc_null/total:.2f}%)", f)
        log(f"  empty string: {desc_empty:,} ({100*desc_empty/total:.2f}%)", f)
        log(f"  valid: {desc_valid:,} ({100*desc_valid/total:.2f}%)", f)
        log("", f)

        # CATEGORIES field
        cat_null = df['CATEGORIES'].isna().sum()
        cat_empty = (df['CATEGORIES'].fillna('') == '').sum() - cat_null
        cat_valid = total - cat_null - cat_empty
        log(f"CATEGORIES field:", f)
        log(f"  null: {cat_null:,} ({100*cat_null/total:.2f}%)", f)
        log(f"  empty string: {cat_empty:,} ({100*cat_empty/total:.2f}%)", f)
        log(f"  valid: {cat_valid:,} ({100*cat_valid/total:.2f}%)", f)
        log("", f)

        # =================================================================
        # SECTION 2: TEXT LENGTH DISTRIBUTIONS
        # =================================================================
        log("=" * 80, f)
        log("SECTION 2: TEXT LENGTH DISTRIBUTIONS", f)
        log("-" * 40, f)
        log("", f)

        # NAME lengths
        log("NAME field lengths:", f)
        df['name_char_count'] = df['NAME'].fillna('').str.len()
        df['name_word_count'] = df['NAME'].fillna('').str.split().str.len().fillna(0).astype(int)

        percentile_stats(df['name_char_count'], "Character count", f)
        log("", f)
        percentile_stats(df['name_word_count'], "Word count", f)
        log("", f)

        # DESCRIPTION lengths
        log("DESCRIPTION field lengths:", f)
        df['desc_char_count'] = df['DESCRIPTION'].fillna('').str.len()
        df['desc_word_count'] = df['DESCRIPTION'].fillna('').str.split().str.len().fillna(0).astype(int)

        percentile_stats(df['desc_char_count'], "Character count", f)
        log("", f)
        percentile_stats(df['desc_word_count'], "Word count", f)
        log("", f)

        # Correlation between NAME and DESCRIPTION lengths
        valid_both = df[(df['name_char_count'] > 0) & (df['desc_char_count'] > 0)]
        if len(valid_both) > 0:
            corr_char = valid_both['name_char_count'].corr(valid_both['desc_char_count'])
            corr_word = valid_both['name_word_count'].corr(valid_both['desc_word_count'])
            log(f"Correlation (NAME vs DESCRIPTION):", f)
            log(f"  char count correlation: {corr_char:.4f}", f)
            log(f"  word count correlation: {corr_word:.4f}", f)
            log(f"  sample size: {len(valid_both):,}", f)
        log("", f)

        # Name equals description rate
        name_eq_desc = (df['NAME'].fillna('') == df['DESCRIPTION'].fillna('')).sum()
        log(f"NAME equals DESCRIPTION exactly: {name_eq_desc:,} ({100*name_eq_desc/total:.2f}%)", f)
        log("", f)

        # =================================================================
        # SECTION 3: CATEGORY PARSING
        # =================================================================
        log("=" * 80, f)
        log("SECTION 3: CATEGORY PARSING", f)
        log("-" * 40, f)
        log("", f)

        log("Parsing CATEGORIES field...", f)
        tqdm.pandas(desc="Parsing categories")
        parsed = df['CATEGORIES'].progress_apply(parse_categories)

        df['brand'] = parsed.apply(lambda x: x['brand'])
        df['colors'] = parsed.apply(lambda x: x['colors'])
        df['sizes'] = parsed.apply(lambda x: x['sizes'])
        df['department_id'] = parsed.apply(lambda x: x['department_id'])
        df['category_ids'] = parsed.apply(lambda x: x['category_ids'])
        df['domain'] = parsed.apply(lambda x: x['domain'])

        log("", f)
        log("Parsed field coverage:", f)

        brand_count = df['brand'].notna().sum()
        log(f"  brand: {brand_count:,} ({100*brand_count/total:.2f}%)", f)

        color_count = df['colors'].apply(lambda x: len(x) > 0).sum()
        log(f"  color (any): {color_count:,} ({100*color_count/total:.2f}%)", f)

        size_count = df['sizes'].apply(lambda x: len(x) > 0).sum()
        log(f"  size (any): {size_count:,} ({100*size_count/total:.2f}%)", f)

        dept_count = df['department_id'].notna().sum()
        log(f"  department_id: {dept_count:,} ({100*dept_count/total:.2f}%)", f)

        cat_id_count = df['category_ids'].apply(lambda x: len(x) > 0).sum()
        log(f"  category_id (any): {cat_id_count:,} ({100*cat_id_count/total:.2f}%)", f)

        domain_count = df['domain'].notna().sum()
        log(f"  domain: {domain_count:,} ({100*domain_count/total:.2f}%)", f)
        log("", f)

        # Domain distribution
        log("Domain distribution:", f)
        domain_dist = df['domain'].value_counts(dropna=False)
        for domain, count in domain_dist.items():
            log(f"  {domain}: {count:,} ({100*count/total:.2f}%)", f)
        log("", f)

        # =================================================================
        # SECTION 4: BRAND ANALYSIS
        # =================================================================
        log("=" * 80, f)
        log("SECTION 4: BRAND ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Brand coverage
        has_brand = df['brand'].notna()
        log(f"Brand coverage:", f)
        log(f"  products with brand: {has_brand.sum():,} ({100*has_brand.mean():.2f}%)", f)
        log(f"  products without brand: {(~has_brand).sum():,} ({100*(~has_brand).mean():.2f}%)", f)
        log("", f)

        # Brand concentration
        brand_counts = df['brand'].value_counts()
        unique_brands = len(brand_counts)
        log(f"Unique brands: {unique_brands:,}", f)

        gini = np.nan
        if unique_brands > 0:
            gini = compute_gini(brand_counts.values)
            log(f"Brand concentration (Gini coefficient): {gini:.4f}", f)
            log("", f)

            # Top 50 brands
            log("Top 50 brands by product count:", f)
            for i, (brand, count) in enumerate(brand_counts.head(50).items(), 1):
                pct = 100 * count / has_brand.sum()
                log(f"  {i:2d}. {brand}: {count:,} ({pct:.2f}%)", f)
            log("", f)

            # Price distribution by top brands
            log("Price distribution by top 20 brands:", f)
            top20_brands = brand_counts.head(20).index.tolist()
            for brand in top20_brands:
                brand_prices = df[df['brand'] == brand][price_col].dropna()
                if len(brand_prices) > 0:
                    log(f"  {brand}:", f)
                    log(f"    n={len(brand_prices):,}, mean=${brand_prices.mean():.2f}, median=${brand_prices.median():.2f}, std=${brand_prices.std():.2f}", f)
        log("", f)

        # =================================================================
        # SECTION 5: COLOR ANALYSIS
        # =================================================================
        log("=" * 80, f)
        log("SECTION 5: COLOR ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        has_color = df['colors'].apply(lambda x: len(x) > 0)
        log(f"Color coverage:", f)
        log(f"  products with color: {has_color.sum():,} ({100*has_color.mean():.2f}%)", f)
        log("", f)

        # Multi-color rate
        multi_color = df['colors'].apply(lambda x: len(x) > 1)
        log(f"Multi-color products: {multi_color.sum():,} ({100*multi_color.mean():.2f}%)", f)

        # Color count distribution
        color_counts_per_product = df['colors'].apply(len)
        log(f"Colors per product distribution:", f)
        for n_colors in range(6):
            count = (color_counts_per_product == n_colors).sum()
            log(f"  {n_colors} colors: {count:,} ({100*count/total:.2f}%)", f)
        log(f"  6+ colors: {(color_counts_per_product >= 6).sum():,} ({100*(color_counts_per_product >= 6).mean():.2f}%)", f)
        log("", f)

        # Flatten colors and count
        all_colors = []
        for colors in df['colors']:
            all_colors.extend(colors)
        color_freq = Counter(all_colors)

        log(f"Unique colors: {len(color_freq):,}", f)
        log("", f)
        log("Top 30 colors by frequency:", f)
        for i, (color, count) in enumerate(color_freq.most_common(30), 1):
            log(f"  {i:2d}. {color}: {count:,}", f)
        log("", f)

        # Price by top colors
        log("Price distribution by top 15 colors:", f)
        top15_colors = [c for c, _ in color_freq.most_common(15)]
        for color in top15_colors:
            color_mask = df['colors'].apply(lambda x: color in x)
            color_prices = df[color_mask][price_col].dropna()
            if len(color_prices) > 0:
                log(f"  {color}: n={len(color_prices):,}, mean=${color_prices.mean():.2f}, median=${color_prices.median():.2f}", f)
        log("", f)

        # =================================================================
        # SECTION 6: SIZE ANALYSIS
        # =================================================================
        log("=" * 80, f)
        log("SECTION 6: SIZE ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        has_size = df['sizes'].apply(lambda x: len(x) > 0)
        log(f"Size coverage:", f)
        log(f"  products with size: {has_size.sum():,} ({100*has_size.mean():.2f}%)", f)
        log("", f)

        # Flatten sizes and count
        all_sizes = []
        for sizes in df['sizes']:
            all_sizes.extend(sizes)
        size_freq = Counter(all_sizes)

        log(f"Unique sizes: {len(size_freq):,}", f)
        log("", f)
        log("Top 30 sizes by frequency:", f)
        for i, (size, count) in enumerate(size_freq.most_common(30), 1):
            log(f"  {i:2d}. {size}: {count:,}", f)
        log("", f)

        # Size standardization analysis
        log("Size standardization analysis:", f)
        standard_sizes = ['us xs', 'us s', 'us m', 'us l', 'us xl', 'us xxl',
                         'us 2xl', 'us 3xl', 'us 4xl', 'us 5xl']
        for std_size in standard_sizes:
            count = size_freq.get(std_size, 0)
            log(f"  {std_size}: {count:,}", f)
        log("", f)

        # Numeric sizes
        numeric_pattern = re.compile(r'^us \d+$')
        numeric_sizes = {s: c for s, c in size_freq.items() if numeric_pattern.match(s)}
        log(f"Numeric sizes (us N pattern): {len(numeric_sizes):,} unique values", f)
        log(f"Top 20 numeric sizes:", f)
        for size, count in sorted(numeric_sizes.items(), key=lambda x: -x[1])[:20]:
            log(f"  {size}: {count:,}", f)
        log("", f)

        # =================================================================
        # SECTION 7: VOCABULARY ANALYSIS (TF-IDF)
        # =================================================================
        log("=" * 80, f)
        log("SECTION 7: VOCABULARY ANALYSIS (TF-IDF)", f)
        log("-" * 40, f)
        log("", f)

        # Prepare text data
        valid_names = df[df['NAME'].notna() & (df['NAME'] != '')]['NAME'].values
        log(f"Valid NAME entries for TF-IDF: {len(valid_names):,}", f)
        log("", f)

        # Unigrams
        log("Fitting TfidfVectorizer (unigrams)...", f)
        vectorizer_uni = TfidfVectorizer(
            max_features=10000,
            min_df=10,
            stop_words='english',
            ngram_range=(1, 1)
        )
        tfidf_uni = vectorizer_uni.fit_transform(valid_names)

        vocab_size_uni = len(vectorizer_uni.vocabulary_)
        log(f"Vocabulary size (unigrams): {vocab_size_uni:,}", f)
        log(f"Matrix shape: {tfidf_uni.shape}", f)
        log(f"Matrix sparsity: {100*(1 - tfidf_uni.nnz / (tfidf_uni.shape[0] * tfidf_uni.shape[1])):.4f}%", f)
        log("", f)

        # Document frequency for unigrams
        doc_freq_uni = np.array((tfidf_uni > 0).sum(axis=0)).flatten()
        feature_names_uni = vectorizer_uni.get_feature_names_out()
        df_sorted_uni = sorted(zip(feature_names_uni, doc_freq_uni), key=lambda x: -x[1])

        log("Top 100 unigrams by document frequency:", f)
        for i, (term, freq) in enumerate(df_sorted_uni[:100], 1):
            pct = 100 * freq / len(valid_names)
            log(f"  {i:3d}. {term}: {freq:,} ({pct:.2f}%)", f)
        log("", f)

        # Bigrams
        log("Fitting TfidfVectorizer (bigrams)...", f)
        vectorizer_bi = TfidfVectorizer(
            max_features=10000,
            min_df=10,
            stop_words='english',
            ngram_range=(2, 2)
        )
        tfidf_bi = vectorizer_bi.fit_transform(valid_names)

        vocab_size_bi = len(vectorizer_bi.vocabulary_)
        log(f"Vocabulary size (bigrams): {vocab_size_bi:,}", f)
        log("", f)

        # Document frequency for bigrams
        doc_freq_bi = np.array((tfidf_bi > 0).sum(axis=0)).flatten()
        feature_names_bi = vectorizer_bi.get_feature_names_out()
        df_sorted_bi = sorted(zip(feature_names_bi, doc_freq_bi), key=lambda x: -x[1])

        log("Top 50 bigrams by document frequency:", f)
        for i, (term, freq) in enumerate(df_sorted_bi[:50], 1):
            pct = 100 * freq / len(valid_names)
            log(f"  {i:2d}. {term}: {freq:,} ({pct:.2f}%)", f)
        log("", f)

        # =================================================================
        # SECTION 8: DESCRIPTION QUALITY
        # =================================================================
        log("=" * 80, f)
        log("SECTION 8: DESCRIPTION QUALITY", f)
        log("-" * 40, f)
        log("", f)

        valid_desc_mask = df['DESCRIPTION'].notna() & (df['DESCRIPTION'] != '')
        valid_desc = df[valid_desc_mask]

        log(f"Products with non-empty DESCRIPTION: {len(valid_desc):,}", f)
        log("", f)

        # Short description rate
        short_desc = (valid_desc['desc_char_count'] < 50).sum()
        log(f"Short descriptions (<50 chars): {short_desc:,} ({100*short_desc/len(valid_desc):.2f}%)", f)

        # Very long description rate
        long_desc = (valid_desc['desc_char_count'] > 1000).sum()
        log(f"Long descriptions (>1000 chars): {long_desc:,} ({100*long_desc/len(valid_desc):.2f}%)", f)

        very_long_desc = (valid_desc['desc_char_count'] > 2000).sum()
        log(f"Very long descriptions (>2000 chars): {very_long_desc:,} ({100*very_long_desc/len(valid_desc):.2f}%)", f)
        log("", f)

        # Description starts with product name rate
        starts_with_name = 0
        for _, row in tqdm(valid_desc.iterrows(), total=len(valid_desc), desc="Checking name in desc"):
            name = str(row['NAME']).lower().strip()
            desc = str(row['DESCRIPTION']).lower().strip()
            if name and desc.startswith(name[:min(20, len(name))]):
                starts_with_name += 1
        log(f"Description starts with NAME: {starts_with_name:,} ({100*starts_with_name/len(valid_desc):.2f}%)", f)
        log("", f)

        # Common description patterns
        log("Common description patterns:", f)
        patterns = {
            'like new': re.compile(r'like new', re.IGNORECASE),
            'brand new': re.compile(r'brand new', re.IGNORECASE),
            'new with tags': re.compile(r'new with tags?|nwt', re.IGNORECASE),
            'free shipping': re.compile(r'free shipping', re.IGNORECASE),
            'size': re.compile(r'\bsize\b', re.IGNORECASE),
            'great condition': re.compile(r'great condition', re.IGNORECASE),
            'excellent condition': re.compile(r'excellent condition', re.IGNORECASE),
            'good condition': re.compile(r'good condition', re.IGNORECASE),
            'pre-owned': re.compile(r'pre-?owned', re.IGNORECASE),
            'vintage': re.compile(r'\bvintage\b', re.IGNORECASE),
            'authentic': re.compile(r'\bauthentic\b', re.IGNORECASE),
            'measurements': re.compile(r'\bmeasurements?\b', re.IGNORECASE),
            'worn once': re.compile(r'worn once', re.IGNORECASE),
            'never worn': re.compile(r'never worn', re.IGNORECASE),
            'smoke free': re.compile(r'smoke[ -]?free', re.IGNORECASE),
            'pet free': re.compile(r'pet[ -]?free', re.IGNORECASE),
        }

        desc_text = valid_desc['DESCRIPTION'].fillna('')
        for pattern_name, pattern in patterns.items():
            matches = desc_text.str.contains(pattern, regex=True).sum()
            pct = 100 * matches / len(valid_desc)
            log(f"  '{pattern_name}': {matches:,} ({pct:.2f}%)", f)
        log("", f)

        # =================================================================
        # SECTION 9: PRICE CORRELATIONS
        # =================================================================
        log("=" * 80, f)
        log("SECTION 9: PRICE CORRELATIONS", f)
        log("-" * 40, f)
        log("", f)

        valid_price = df[price_col].notna() & (df[price_col] > 0)
        log(f"Products with valid price (>0): {valid_price.sum():,} ({100*valid_price.mean():.2f}%)", f)
        log("", f)

        # Price distribution
        prices = df[valid_price][price_col]
        percentile_stats(prices, "Price distribution", f)
        log("", f)

        # Price vs NAME length
        price_name_data = df[valid_price & (df['name_char_count'] > 0)]
        if len(price_name_data) > 0:
            corr_price_name = price_name_data[price_col].corr(price_name_data['name_char_count'])
            corr_price_name_word = price_name_data[price_col].corr(price_name_data['name_word_count'])
            log(f"Price vs NAME length:", f)
            log(f"  correlation (char count): {corr_price_name:.4f}", f)
            log(f"  correlation (word count): {corr_price_name_word:.4f}", f)
            log(f"  sample size: {len(price_name_data):,}", f)
        log("", f)

        # Price vs DESCRIPTION length
        price_desc_data = df[valid_price & (df['desc_char_count'] > 0)]
        if len(price_desc_data) > 0:
            corr_price_desc = price_desc_data[price_col].corr(price_desc_data['desc_char_count'])
            corr_price_desc_word = price_desc_data[price_col].corr(price_desc_data['desc_word_count'])
            log(f"Price vs DESCRIPTION length:", f)
            log(f"  correlation (char count): {corr_price_desc:.4f}", f)
            log(f"  correlation (word count): {corr_price_desc_word:.4f}", f)
            log(f"  sample size: {len(price_desc_data):,}", f)
        log("", f)

        # Price by brand presence
        price_with_brand = df[valid_price & df['brand'].notna()][price_col]
        price_without_brand = df[valid_price & df['brand'].isna()][price_col]
        log(f"Price by brand presence:", f)
        log(f"  with brand: n={len(price_with_brand):,}, mean=${price_with_brand.mean():.2f}, median=${price_with_brand.median():.2f}", f)
        log(f"  without brand: n={len(price_without_brand):,}, mean=${price_without_brand.mean():.2f}, median=${price_without_brand.median():.2f}", f)
        log("", f)

        # Price outliers
        log("Price outliers:", f)
        high_price = (prices > 10000).sum()
        very_high_price = (prices > 50000).sum()
        extreme_price = (prices > 100000).sum()
        log(f"  >$10,000: {high_price:,} ({100*high_price/len(prices):.4f}%)", f)
        log(f"  >$50,000: {very_high_price:,} ({100*very_high_price/len(prices):.4f}%)", f)
        log(f"  >$100,000: {extreme_price:,} ({100*extreme_price/len(prices):.4f}%)", f)

        zero_price = (df[price_col] == 0).sum()
        negative_price = (df[price_col] < 0).sum()
        log(f"  price = 0: {zero_price:,} ({100*zero_price/total:.4f}%)", f)
        log(f"  price < 0: {negative_price:,} ({100*negative_price/total:.4f}%)", f)
        log("", f)

        # Top 10 highest priced products
        log("Top 10 highest priced products:", f)
        top_prices = df.nlargest(10, price_col)[['PRODUCT_ID', 'NAME', price_col, 'brand']]
        for _, row in top_prices.iterrows():
            name_trunc = str(row['NAME'])[:60] if pd.notna(row['NAME']) else 'N/A'
            log(f"  ${row[price_col]:,.0f}: {name_trunc}... (brand: {row['brand']})", f)
        log("", f)

        # =================================================================
        # SECTION 10: SUMMARY
        # =================================================================
        log("=" * 80, f)
        log("SECTION 10: SUMMARY", f)
        log("-" * 40, f)
        log("", f)

        summary = [
            ("Total products", f"{total:,}"),
            ("NAME coverage", f"{100*name_valid/total:.1f}%"),
            ("DESCRIPTION coverage", f"{100*desc_valid/total:.1f}%"),
            ("CATEGORIES coverage", f"{100*cat_valid/total:.1f}%"),
            ("Brand coverage", f"{100*brand_count/total:.1f}%"),
            ("Color coverage", f"{100*color_count/total:.1f}%"),
            ("Size coverage", f"{100*size_count/total:.1f}%"),
            ("Valid price coverage", f"{100*valid_price.sum()/total:.1f}%"),
            ("NAME = DESCRIPTION rate", f"{100*name_eq_desc/total:.1f}%"),
            ("Unique brands", f"{unique_brands:,}"),
            ("Unique colors", f"{len(color_freq):,}"),
            ("Unique sizes", f"{len(size_freq):,}"),
            ("Unigram vocabulary", f"{vocab_size_uni:,}"),
            ("Bigram vocabulary", f"{vocab_size_bi:,}"),
            ("Median NAME length (words)", f"{df['name_word_count'].median():.0f}"),
            ("Median DESCRIPTION length (words)", f"{df['desc_word_count'].median():.0f}"),
            ("Median price", f"${prices.median():.2f}"),
            ("Mean price", f"${prices.mean():.2f}"),
            ("Brand concentration (Gini)", f"{gini:.4f}"),
        ]

        for metric, value in summary:
            log(f"  {metric}: {value}", f)
        log("", f)

        log("=" * 80, f)
        log("END OF CATALOG NLP EDA", f)
        log("=" * 80, f)

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
