#!/usr/bin/env python3
"""
CTR Prediction with NLP Features
================================
Extends baseline CTR model with text features from product catalog.

Features:
- Position features (7): position, log_position, relative_position, row, col, is_top_row, is_first_col
- Auction features (8): quality, log_quality, relative_quality, quality_rank, bid, log_bid, relative_bid, bid_rank
- NLP text features (~60): text stats, TF-IDF embeddings
- Structured catalog features (~25): brand, color, size, price
- Interaction features (6): position x quality, etc.
- Session context (6): n_items, placement one-hot
"""

import sys
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path('/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/0_data/round1')
CATALOG_DIR = DATA_DIR
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / '17_ctr_nlp_features.txt'

# Config
TFIDF_DIMS = 50
TOP_COLORS = ['black', 'blue', 'white', 'pink', 'gray', 'grey', 'red', 'brown', 'green', 'cream', 'tan', 'purple', 'navy', 'beige']
TOP_BRAND_N = 50
RANDOM_STATE = 42


class OutputCapture:
    """Capture all output to file and stdout."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def parse_categories(cat_str):
    """Parse CATEGORIES JSON string into dict of extracted features."""
    result = {
        'brand': None,
        'colors': [],
        'sizes': [],
        'department': None,
        'category_ids': []
    }

    if pd.isna(cat_str) or not cat_str:
        return result

    try:
        items = json.loads(cat_str)
        for item in items:
            if '#' in item:
                key, val = item.split('#', 1)
                key = key.lower().strip()
                val = val.lower().strip()

                if key == 'brand':
                    result['brand'] = val
                elif key == 'color':
                    result['colors'].append(val)
                elif key == 'size':
                    result['sizes'].append(val)
                elif key == 'department':
                    result['department'] = val
                elif key == 'category':
                    result['category_ids'].append(val)
    except (json.JSONDecodeError, ValueError):
        pass

    return result


def extract_text_features(df):
    """Extract text statistics from NAME and DESCRIPTION."""
    print("\n--- Extracting Text Statistics ---")

    features = pd.DataFrame(index=df.index)

    # Name features
    name = df['NAME'].fillna('')
    features['name_length'] = name.str.len()
    features['name_word_count'] = name.str.split().str.len().fillna(0)
    features['name_caps_ratio'] = name.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

    # Special keywords in name
    name_lower = name.str.lower()
    features['has_nwt'] = name_lower.str.contains(r'\bnwt\b|\bnew with tags\b', regex=True).astype(int)
    features['has_vintage'] = name_lower.str.contains(r'\bvintage\b', regex=True).astype(int)
    features['has_size_in_name'] = name_lower.str.contains(r'\bsize\s*\d|sz\s*\d|\b(xs|s|m|l|xl|xxl)\b', regex=True).astype(int)
    features['has_brand_in_name'] = name_lower.str.contains(r'\bnike\b|\badidas\b|\bgucci\b|\bprada\b|\bcoach\b|\bzara\b', regex=True).astype(int)
    features['has_free_shipping'] = name_lower.str.contains(r'\bfree ship', regex=True).astype(int)

    # Description features
    desc = df['DESCRIPTION'].fillna('')
    features['desc_length'] = desc.str.len()
    features['desc_word_count'] = desc.str.split().str.len().fillna(0)
    features['desc_equals_name'] = (name == desc).astype(int)

    print(f"  Text stat features: {features.shape[1]}")

    return features


def extract_tfidf_features(df, n_components=TFIDF_DIMS):
    """Extract TF-IDF features from combined NAME + DESCRIPTION."""
    print("\n--- Extracting TF-IDF Features ---")

    # Combine name and description
    text = (df['NAME'].fillna('') + ' ' + df['DESCRIPTION'].fillna('')).str.lower()
    text = text.str.replace(r'[^\w\s]', ' ', regex=True)

    print(f"  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(tqdm(text, desc="  TF-IDF"))
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Reduce dimensions with SVD
    print(f"  Reducing to {n_components} dimensions with SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    # Create DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_reduced,
        columns=[f'tfidf_{i}' for i in range(n_components)],
        index=df.index
    )

    # Store vocabulary info for later analysis
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_df, vectorizer, svd, feature_names


def extract_structured_features(df, brand_freq_map, top_brands):
    """Extract structured features from CATEGORIES."""
    print("\n--- Extracting Structured Catalog Features ---")

    features = pd.DataFrame(index=df.index)

    # Parse categories
    print("  Parsing CATEGORIES JSON...")
    parsed = [parse_categories(c) for c in tqdm(df['CATEGORIES'], desc="  Parsing")]

    # Brand features
    brands = [p['brand'] for p in parsed]
    features['has_brand'] = pd.Series([1 if b else 0 for b in brands], index=df.index)
    features['brand_frequency'] = pd.Series(
        [np.log1p(brand_freq_map.get(b, 0)) if b else 0 for b in brands],
        index=df.index
    )
    features['is_top_brand'] = pd.Series(
        [1 if b in top_brands else 0 for b in brands],
        index=df.index
    )

    # Color features
    colors_list = [p['colors'] for p in parsed]
    features['n_colors'] = pd.Series([len(c) for c in colors_list], index=df.index)

    for color in TOP_COLORS:
        features[f'color_{color}'] = pd.Series(
            [1 if color in c or (color == 'gray' and 'grey' in c) else 0 for c in colors_list],
            index=df.index
        )

    # Size features
    sizes_list = [p['sizes'] for p in parsed]
    features['has_size'] = pd.Series([1 if s else 0 for s in sizes_list], index=df.index)
    features['is_one_size'] = pd.Series(
        [1 if any('os' in sz or 'one size' in sz for sz in s) else 0 for s in sizes_list],
        index=df.index
    )

    # Price features (from catalog)
    price = df['CATALOG_PRICE'].fillna(df['CATALOG_PRICE'].median())
    features['catalog_price'] = price
    features['log_price'] = np.log1p(price)
    features['price_bucket'] = pd.qcut(price, q=4, labels=False, duplicates='drop')

    print(f"  Structured features: {features.shape[1]}")

    return features


def build_position_features(df):
    """Build position-related features."""
    print("\n--- Building Position Features ---")

    features = pd.DataFrame(index=df.index)

    # Basic position
    features['position'] = df['position']
    features['log_position'] = np.log1p(df['position'])

    # Relative position within session
    features['relative_position'] = df.groupby('auction_id')['position'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # Grid layout features (assuming 4-column grid)
    features['row'] = (df['position'] - 1) // 4 + 1
    features['col'] = (df['position'] - 1) % 4 + 1
    features['is_top_row'] = (features['row'] == 1).astype(int)
    features['is_first_col'] = (features['col'] == 1).astype(int)

    print(f"  Position features: {features.shape[1]}")

    return features


def build_auction_features(df):
    """Build auction/bid-related features."""
    print("\n--- Building Auction Features ---")

    features = pd.DataFrame(index=df.index)

    # Quality features
    features['quality'] = df['quality']
    features['log_quality'] = np.log1p(df['quality'])
    features['relative_quality'] = df.groupby('auction_id')['quality'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )
    features['quality_rank'] = df.groupby('auction_id')['quality'].rank(ascending=False)

    # Bid features
    features['bid'] = df['bid']
    features['log_bid'] = np.log1p(df['bid'])
    features['relative_bid'] = df.groupby('auction_id')['bid'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )
    features['bid_rank'] = df.groupby('auction_id')['bid'].rank(ascending=False)

    print(f"  Auction features: {features.shape[1]}")

    return features


def build_interaction_features(pos_feat, auction_feat):
    """Build interaction features between position and auction metrics."""
    print("\n--- Building Interaction Features ---")

    features = pd.DataFrame(index=pos_feat.index)

    # Position x Quality/Bid interactions
    features['position_x_quality'] = pos_feat['position'] * auction_feat['quality']
    features['position_x_bid'] = pos_feat['position'] * auction_feat['bid']
    features['quality_x_bid'] = auction_feat['quality'] * auction_feat['bid']

    # Log interactions
    features['log_pos_x_log_quality'] = pos_feat['log_position'] * auction_feat['log_quality']

    # Ratios
    features['quality_bid_ratio'] = auction_feat['quality'] / (auction_feat['bid'] + 1)
    features['position_quality_ratio'] = pos_feat['position'] / (auction_feat['quality'] + 0.01)

    print(f"  Interaction features: {features.shape[1]}")

    return features


def build_session_features(df):
    """Build session context features."""
    print("\n--- Building Session Context Features ---")

    features = pd.DataFrame(index=df.index)

    # Session size
    features['n_items'] = df['n_items']
    features['log_n_items'] = np.log1p(df['n_items'])

    # Placement one-hot (excluding n_clicks to avoid leakage)
    placement_dummies = pd.get_dummies(df['placement'], prefix='placement')
    features = pd.concat([features, placement_dummies], axis=1)

    print(f"  Session context features: {features.shape[1]}")

    return features


def train_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate the CTR prediction model."""
    print("\n--- Training Model ---")

    if USE_XGBOOST:
        print("  Using XGBoost")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
    else:
        print("  Using sklearn GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE
        )

    print(f"  Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    print(f"\n  Test AUC-ROC: {auc_score:.4f}")
    print(f"  Test Log Loss: {logloss:.4f}")
    print(f"  Test PR-AUC: {pr_auc:.4f}")

    # Feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return model, {
        'auc': auc_score,
        'logloss': logloss,
        'pr_auc': pr_auc
    }, importance_df


def main():
    output = OutputCapture(OUTPUT_PATH)
    sys.stdout = output

    print("=" * 80)
    print("CTR PREDICTION WITH NLP FEATURES")
    print("=" * 80)
    print(f"\nRun timestamp: {datetime.now().isoformat()}")
    print(f"Output file: {OUTPUT_PATH}")

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. DATA LOADING")
    print("=" * 80)

    print("\n--- Loading Session Items ---")
    session_items = pd.read_parquet(DATA_DIR / 'session_items.parquet')
    print(f"  Shape: {session_items.shape}")
    print(f"  Columns: {list(session_items.columns)}")
    print(f"  CTR: {session_items.clicked.mean():.4f} ({session_items.clicked.sum()} clicks)")

    print("\n--- Loading Catalog ---")
    catalog = pd.read_parquet(CATALOG_DIR / 'catalog_all.parquet')
    print(f"  Shape: {catalog.shape}")
    print(f"  Columns: {list(catalog.columns)}")

    # -------------------------------------------------------------------------
    # 2. Join Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. DATA JOIN")
    print("=" * 80)

    print("\n--- Joining session_items with catalog ---")
    n_before = len(session_items)

    # Merge
    df = session_items.merge(
        catalog[['PRODUCT_ID', 'NAME', 'DESCRIPTION', 'CATEGORIES', 'CATALOG_PRICE']],
        left_on='product_id',
        right_on='PRODUCT_ID',
        how='left'
    )

    n_matched = df['PRODUCT_ID'].notna().sum()
    print(f"  Rows before join: {n_before}")
    print(f"  Rows matched: {n_matched} ({n_matched/n_before*100:.2f}%)")
    print(f"  Rows unmatched: {n_before - n_matched}")

    # Fill missing text with empty strings
    df['NAME'] = df['NAME'].fillna('')
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
    df['CATEGORIES'] = df['CATEGORIES'].fillna('[]')
    df['CATALOG_PRICE'] = df['CATALOG_PRICE'].fillna(df['CATALOG_PRICE'].median())

    # -------------------------------------------------------------------------
    # 3. Pre-compute Brand Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. BRAND STATISTICS")
    print("=" * 80)

    print("\n--- Computing brand frequencies from catalog ---")
    all_brands = []
    for cat_str in tqdm(catalog['CATEGORIES'], desc="  Extracting brands"):
        parsed = parse_categories(cat_str)
        if parsed['brand']:
            all_brands.append(parsed['brand'])

    brand_counts = pd.Series(all_brands).value_counts()
    brand_freq_map = brand_counts.to_dict()
    top_brands = set(brand_counts.head(TOP_BRAND_N).index)

    print(f"  Unique brands: {len(brand_counts)}")
    print(f"  Top {TOP_BRAND_N} brands account for {brand_counts.head(TOP_BRAND_N).sum() / len(all_brands) * 100:.2f}% of branded products")
    print(f"\n  Top 10 brands:")
    for i, (brand, count) in enumerate(brand_counts.head(10).items()):
        print(f"    {i+1}. {brand}: {count}")

    # -------------------------------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. FEATURE ENGINEERING")
    print("=" * 80)

    # Position features
    pos_features = build_position_features(df)

    # Auction features
    auction_features = build_auction_features(df)

    # Session features
    session_features = build_session_features(df)

    # Text features
    text_features = extract_text_features(df)

    # TF-IDF features
    tfidf_features, vectorizer, svd, vocab = extract_tfidf_features(df)

    # Structured catalog features
    structured_features = extract_structured_features(df, brand_freq_map, top_brands)

    # Interaction features
    interaction_features = build_interaction_features(pos_features, auction_features)

    # Price relative to session
    print("\n--- Adding session-relative price ---")
    session_avg_price = df.groupby('auction_id')['CATALOG_PRICE'].transform('mean')
    structured_features['relative_price_in_session'] = df['CATALOG_PRICE'] / (session_avg_price + 1)

    # -------------------------------------------------------------------------
    # 5. Combine All Features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. FEATURE MATRIX")
    print("=" * 80)

    all_features = pd.concat([
        pos_features,
        auction_features,
        session_features,
        text_features,
        tfidf_features,
        structured_features,
        interaction_features
    ], axis=1)

    print(f"\n  Total features: {all_features.shape[1]}")
    print(f"  Total samples: {all_features.shape[0]}")

    # Feature categories
    feature_categories = {
        'Position': list(pos_features.columns),
        'Auction': list(auction_features.columns),
        'Session': list(session_features.columns),
        'Text Stats': list(text_features.columns),
        'TF-IDF': list(tfidf_features.columns),
        'Structured': list(structured_features.columns),
        'Interaction': list(interaction_features.columns)
    }

    print("\n  Features by category:")
    for cat, feats in feature_categories.items():
        print(f"    {cat}: {len(feats)}")

    print("\n  All feature names:")
    for i, col in enumerate(all_features.columns):
        col_data = all_features[col].astype(float)
        print(f"    {i+1:3d}. {col:40s} mean={col_data.mean():10.4f}  std={col_data.std():10.4f}  min={col_data.min():10.4f}  max={col_data.max():10.4f}")

    # -------------------------------------------------------------------------
    # 6. NLP Feature Examples
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. NLP FEATURE EXAMPLES")
    print("=" * 80)

    # Sample products with clicks
    clicked_idx = df[df['clicked'] == 1].sample(5, random_state=RANDOM_STATE).index
    not_clicked_idx = df[df['clicked'] == 0].sample(5, random_state=RANDOM_STATE).index

    print("\n--- Clicked Products ---")
    for idx in clicked_idx:
        print(f"\n  Product: {df.loc[idx, 'product_id']}")
        print(f"  NAME: {df.loc[idx, 'NAME'][:100]}...")
        print(f"  Position: {df.loc[idx, 'position']}, Quality: {df.loc[idx, 'quality']:.4f}")
        print(f"  name_length: {text_features.loc[idx, 'name_length']:.0f}")
        print(f"  has_nwt: {text_features.loc[idx, 'has_nwt']}")
        print(f"  has_brand: {structured_features.loc[idx, 'has_brand']}")
        print(f"  log_price: {structured_features.loc[idx, 'log_price']:.2f}")

    print("\n--- Not Clicked Products ---")
    for idx in not_clicked_idx:
        print(f"\n  Product: {df.loc[idx, 'product_id']}")
        print(f"  NAME: {df.loc[idx, 'NAME'][:100]}...")
        print(f"  Position: {df.loc[idx, 'position']}, Quality: {df.loc[idx, 'quality']:.4f}")
        print(f"  name_length: {text_features.loc[idx, 'name_length']:.0f}")
        print(f"  has_nwt: {text_features.loc[idx, 'has_nwt']}")
        print(f"  has_brand: {structured_features.loc[idx, 'has_brand']}")
        print(f"  log_price: {structured_features.loc[idx, 'log_price']:.2f}")

    # -------------------------------------------------------------------------
    # 7. Train/Test Split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. TRAIN/TEST SPLIT")
    print("=" * 80)

    # Split by auction_id to avoid leakage
    auction_ids = df['auction_id'].unique()
    train_auctions, test_auctions = train_test_split(
        auction_ids, test_size=0.2, random_state=RANDOM_STATE
    )

    train_mask = df['auction_id'].isin(train_auctions)
    test_mask = df['auction_id'].isin(test_auctions)

    X_train = all_features[train_mask].values
    X_test = all_features[test_mask].values
    y_train = df.loc[train_mask, 'clicked'].values
    y_test = df.loc[test_mask, 'clicked'].values

    print(f"\n  Train auctions: {len(train_auctions)}")
    print(f"  Test auctions: {len(test_auctions)}")
    print(f"  Train samples: {len(X_train)} (CTR: {y_train.mean():.4f})")
    print(f"  Test samples: {len(X_test)} (CTR: {y_test.mean():.4f})")

    # -------------------------------------------------------------------------
    # 8. Model Comparison: Baseline vs NLP-enhanced
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. MODEL COMPARISON")
    print("=" * 80)

    # Baseline features (no NLP)
    baseline_cols = list(pos_features.columns) + list(auction_features.columns) + list(session_features.columns)
    baseline_idx = [list(all_features.columns).index(c) for c in baseline_cols]

    X_train_baseline = X_train[:, baseline_idx]
    X_test_baseline = X_test[:, baseline_idx]

    print("\n--- Baseline Model (Position + Auction + Session) ---")
    print(f"  Features: {len(baseline_cols)}")
    baseline_model, baseline_metrics, baseline_importance = train_evaluate_model(
        X_train_baseline, X_test_baseline, y_train, y_test, baseline_cols
    )

    print("\n--- Full Model (with NLP features) ---")
    print(f"  Features: {all_features.shape[1]}")
    full_model, full_metrics, full_importance = train_evaluate_model(
        X_train, X_test, y_train, y_test, list(all_features.columns)
    )

    # -------------------------------------------------------------------------
    # 9. Results Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("9. RESULTS COMPARISON")
    print("=" * 80)

    print("\n  Metric            Baseline    Full Model    Improvement")
    print("  " + "-" * 55)
    print(f"  AUC-ROC           {baseline_metrics['auc']:.4f}      {full_metrics['auc']:.4f}        {(full_metrics['auc'] - baseline_metrics['auc'])*100:+.2f}%")
    print(f"  Log Loss          {baseline_metrics['logloss']:.4f}      {full_metrics['logloss']:.4f}        {(baseline_metrics['logloss'] - full_metrics['logloss'])*100:+.2f}%")
    print(f"  PR-AUC            {baseline_metrics['pr_auc']:.4f}      {full_metrics['pr_auc']:.4f}        {(full_metrics['pr_auc'] - baseline_metrics['pr_auc'])*100:+.2f}%")

    # -------------------------------------------------------------------------
    # 10. Feature Importances
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. FEATURE IMPORTANCES (Full Model)")
    print("=" * 80)

    print("\n--- Top 30 Features ---")
    for i, row in full_importance.head(30).iterrows():
        print(f"  {full_importance.index.get_loc(i)+1:3d}. {row['feature']:40s} {row['importance']:.6f}")

    print("\n--- All Features (sorted) ---")
    for i, row in full_importance.iterrows():
        print(f"  {full_importance.index.get_loc(i)+1:3d}. {row['feature']:40s} {row['importance']:.6f}")

    # -------------------------------------------------------------------------
    # 11. Category-level Importance Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("11. CATEGORY-LEVEL IMPORTANCE")
    print("=" * 80)

    importance_by_category = {}
    for cat, feats in feature_categories.items():
        cat_importance = full_importance[full_importance['feature'].isin(feats)]['importance'].sum()
        importance_by_category[cat] = cat_importance

    total_importance = sum(importance_by_category.values())
    print("\n  Category             Total Importance    % of Total")
    print("  " + "-" * 55)
    for cat, imp in sorted(importance_by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {imp:.6f}            {imp/total_importance*100:.2f}%")

    # -------------------------------------------------------------------------
    # 12. NLP Feature Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("12. NLP FEATURE ANALYSIS")
    print("=" * 80)

    nlp_features = list(text_features.columns) + list(tfidf_features.columns) + list(structured_features.columns)
    nlp_importance = full_importance[full_importance['feature'].isin(nlp_features)]

    print("\n--- Top 20 NLP Features by Importance ---")
    for i, row in nlp_importance.head(20).iterrows():
        print(f"  {nlp_importance.index.get_loc(i)+1:3d}. {row['feature']:40s} {row['importance']:.6f}")

    # Text stats vs TF-IDF
    text_stat_imp = full_importance[full_importance['feature'].isin(text_features.columns)]['importance'].sum()
    tfidf_imp = full_importance[full_importance['feature'].isin(tfidf_features.columns)]['importance'].sum()
    struct_imp = full_importance[full_importance['feature'].isin(structured_features.columns)]['importance'].sum()

    print("\n--- NLP Sub-category Importance ---")
    print(f"  Text Statistics:     {text_stat_imp:.6f} ({text_stat_imp/total_importance*100:.2f}%)")
    print(f"  TF-IDF Embeddings:   {tfidf_imp:.6f} ({tfidf_imp/total_importance*100:.2f}%)")
    print(f"  Structured Features: {struct_imp:.6f} ({struct_imp/total_importance*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 13. Click Rate by Feature Buckets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("13. CLICK RATE BY FEATURE BUCKETS")
    print("=" * 80)

    # Position
    print("\n--- CTR by Position ---")
    pos_ctr = df.groupby('position')['clicked'].agg(['mean', 'count']).head(20)
    for pos, row in pos_ctr.iterrows():
        print(f"  Position {int(pos):2d}: CTR={row['mean']:.4f} (n={int(row['count']):5d})")

    # Has brand
    print("\n--- CTR by Has Brand ---")
    brand_ctr = df.groupby(structured_features['has_brand'])['clicked'].agg(['mean', 'count'])
    for has_brand, row in brand_ctr.iterrows():
        print(f"  has_brand={int(has_brand)}: CTR={row['mean']:.4f} (n={int(row['count']):6d})")

    # Name length buckets
    print("\n--- CTR by Name Length Bucket ---")
    df_temp = df.copy()
    df_temp['name_length_bucket'] = pd.cut(text_features['name_length'], bins=[0, 30, 50, 80, 150, 500], labels=['<30', '30-50', '50-80', '80-150', '>150'])
    name_ctr = df_temp.groupby('name_length_bucket')['clicked'].agg(['mean', 'count'])
    for bucket, row in name_ctr.iterrows():
        print(f"  {bucket}: CTR={row['mean']:.4f} (n={int(row['count']):6d})")

    # Price buckets
    print("\n--- CTR by Price Bucket ---")
    df_temp['price_bucket_label'] = pd.cut(df['CATALOG_PRICE'], bins=[0, 20, 50, 100, 500, 10000], labels=['<$20', '$20-50', '$50-100', '$100-500', '>$500'])
    price_ctr = df_temp.groupby('price_bucket_label')['clicked'].agg(['mean', 'count'])
    for bucket, row in price_ctr.iterrows():
        print(f"  {bucket}: CTR={row['mean']:.4f} (n={int(row['count']):6d})")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
Dataset:
  - Total samples: {len(df)}
  - Unique auctions: {df['auction_id'].nunique()}
  - Overall CTR: {df['clicked'].mean():.4f}
  - Catalog match rate: {n_matched/n_before*100:.2f}%

Model Performance:
  - Baseline AUC: {baseline_metrics['auc']:.4f} ({len(baseline_cols)} features)
  - Full Model AUC: {full_metrics['auc']:.4f} ({all_features.shape[1]} features)
  - Improvement: {(full_metrics['auc'] - baseline_metrics['auc'])*100:+.2f}%

Top Feature Categories by Importance:
""")
    for cat, imp in sorted(importance_by_category.items(), key=lambda x: -x[1])[:5]:
        print(f"  - {cat}: {imp/total_importance*100:.1f}%")

    print(f"""
Key NLP Findings:
  - Text Statistics importance: {text_stat_imp/total_importance*100:.2f}%
  - TF-IDF embeddings importance: {tfidf_imp/total_importance*100:.2f}%
  - Structured catalog features importance: {struct_imp/total_importance*100:.2f}%

Top 5 individual features:
""")
    for i, row in full_importance.head(5).iterrows():
        print(f"  {full_importance.index.get_loc(i)+1}. {row['feature']}: {row['importance']:.4f}")

    print("\n" + "=" * 80)
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 80)

    output.close()
    sys.stdout = output.stdout


if __name__ == '__main__':
    main()
