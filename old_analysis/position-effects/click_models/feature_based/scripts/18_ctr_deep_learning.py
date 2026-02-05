#!/usr/bin/env python3
"""
CTR Prediction with Deep Learning (DeepFM-style)
=================================================
Upgrades from classical ML (XGBoost + TF-IDF) to modern deep learning.

Architecture:
- Sparse Feature Embeddings: position, placement, brand (learned)
- Dense Features: quality, bid, price, session context (normalized)
- Product Text: Pre-computed embeddings (50-dim) or SentenceTransformer (384-dim)
- FM Layer: Learns pairwise interactions between all embedding pairs
- Deep Layer: MLP learns higher-order interactions

Baseline (v17 XGBoost): AUC 0.6007 with 112 features
"""

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path('/Users/pranjal/Code/topsort-incrementality/analysis/position-effects/0_data/round1')
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / '18_ctr_deep_learning.txt'

# Config
RANDOM_STATE = 42
TOP_BRAND_N = 500
BATCH_SIZE = 256
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM_POSITION = 16
EMBED_DIM_PLACEMENT = 8
EMBED_DIM_BRAND = 32
MLP_DIMS = [256, 128, 64]
DROPOUT = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


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
    result = {'brand': None}
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
                    break
    except (json.JSONDecodeError, ValueError):
        pass
    return result


class CTRDataset(Dataset):
    """PyTorch Dataset for CTR prediction."""

    def __init__(self, sparse_features, dense_features, product_embeddings, labels):
        self.sparse_features = torch.LongTensor(sparse_features)
        self.dense_features = torch.FloatTensor(dense_features)
        self.product_embeddings = torch.FloatTensor(product_embeddings)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sparse_features[idx],
            self.dense_features[idx],
            self.product_embeddings[idx],
            self.labels[idx]
        )


class FMLayer(nn.Module):
    """Factorization Machine layer for pairwise interactions."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.V = nn.Parameter(torch.randn(input_dim, embed_dim) * 0.01)

    def forward(self, x):
        # x: (batch, input_dim)
        # FM formula: 0.5 * sum((sum(v_i * x_i))^2 - sum(v_i^2 * x_i^2))
        square_of_sum = torch.pow(torch.mm(x, self.V), 2)
        sum_of_square = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return output


class DeepFMCTR(nn.Module):
    """DeepFM-style CTR prediction model."""

    def __init__(self,
                 n_positions=65,
                 n_placements=6,
                 n_brands=502,
                 product_embed_dim=50,
                 n_dense_features=8,
                 embed_dim_position=16,
                 embed_dim_placement=8,
                 embed_dim_brand=32,
                 mlp_dims=[256, 128, 64],
                 dropout=0.2,
                 use_fm=True):
        super().__init__()

        self.use_fm = use_fm

        # Embedding layers for sparse features
        self.embed_position = nn.Embedding(n_positions, embed_dim_position)
        self.embed_placement = nn.Embedding(n_placements, embed_dim_placement)
        self.embed_brand = nn.Embedding(n_brands, embed_dim_brand)

        # Total embedding dimension
        total_embed_dim = embed_dim_position + embed_dim_placement + embed_dim_brand + product_embed_dim
        total_input_dim = total_embed_dim + n_dense_features

        # FM layer for pairwise interactions
        if use_fm:
            self.fm_layer = FMLayer(total_input_dim, 10)

        # Deep layers (MLP)
        layers = []
        input_dim = total_input_dim
        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*layers)

        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sparse_features, dense_features, product_embeddings):
        # sparse_features: (batch, 3) - [position_idx, placement_idx, brand_idx]
        position_embed = self.embed_position(sparse_features[:, 0])
        placement_embed = self.embed_placement(sparse_features[:, 1])
        brand_embed = self.embed_brand(sparse_features[:, 2])

        # Concatenate all embeddings
        all_embeddings = torch.cat([
            position_embed,
            placement_embed,
            brand_embed,
            product_embeddings
        ], dim=1)

        # Concatenate with dense features
        x = torch.cat([all_embeddings, dense_features], dim=1)

        # Deep component
        deep_out = self.deep_layers(x)

        # FM component (optional)
        if self.use_fm:
            fm_out = self.fm_layer(x)
            output = deep_out + fm_out + self.bias
        else:
            output = deep_out + self.bias

        return torch.sigmoid(output).squeeze(1)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for sparse, dense, product_emb, labels in dataloader:
        sparse = sparse.to(device)
        dense = dense.to(device)
        product_emb = product_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sparse, dense, product_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sparse, dense, product_emb, labels in dataloader:
            sparse = sparse.to(device)
            dense = dense.to(device)
            product_emb = product_emb.to(device)
            labels = labels.to(device)

            outputs = model(sparse, dense, product_emb)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_batches
    auc_score = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc_score, np.array(all_preds), np.array(all_labels)


def main():
    output = OutputCapture(OUTPUT_PATH)
    sys.stdout = output

    print("=" * 80)
    print("CTR PREDICTION WITH DEEP LEARNING (DeepFM-style)")
    print("=" * 80)
    print(f"\nRun timestamp: {datetime.now().isoformat()}")
    print(f"Output file: {OUTPUT_PATH}")

    # -------------------------------------------------------------------------
    # 1. Environment Check
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. ENVIRONMENT CHECK")
    print("=" * 80)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon GPU)")
        # Test MPS
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x)
            print("MPS test: PASSED")
        except Exception as e:
            print(f"MPS test failed: {e}")
            device = torch.device('cpu')
            print("Falling back to CPU")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. DATA LOADING")
    print("=" * 80)

    print("\n--- Loading Session Items ---")
    session_items = pd.read_parquet(DATA_DIR / 'session_items.parquet')
    print(f"  Shape: {session_items.shape}")
    print(f"  Columns: {list(session_items.columns)}")
    print(f"  CTR: {session_items.clicked.mean():.4f} ({session_items.clicked.sum()} clicks)")

    print("\n--- Loading Catalog ---")
    catalog = pd.read_parquet(DATA_DIR / 'catalog_all.parquet')
    print(f"  Shape: {catalog.shape}")

    print("\n--- Loading Pre-computed Product Embeddings ---")
    product_embeddings_all = np.load(DATA_DIR / 'product_embeddings.npy')
    print(f"  Shape: {product_embeddings_all.shape}")

    # Create product_id to index mapping from catalog
    product_id_to_idx = {pid: idx for idx, pid in enumerate(catalog['PRODUCT_ID'])}
    print(f"  Unique products in catalog: {len(product_id_to_idx)}")

    # -------------------------------------------------------------------------
    # 3. Join Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. DATA JOIN")
    print("=" * 80)

    print("\n--- Joining session_items with catalog ---")
    n_before = len(session_items)

    df = session_items.merge(
        catalog[['PRODUCT_ID', 'NAME', 'CATEGORIES', 'CATALOG_PRICE']],
        left_on='product_id',
        right_on='PRODUCT_ID',
        how='left'
    )

    n_matched = df['PRODUCT_ID'].notna().sum()
    print(f"  Rows before join: {n_before}")
    print(f"  Rows matched: {n_matched} ({n_matched/n_before*100:.2f}%)")

    # Fill missing
    df['CATALOG_PRICE'] = df['CATALOG_PRICE'].fillna(df['CATALOG_PRICE'].median())
    df['CATEGORIES'] = df['CATEGORIES'].fillna('[]')

    # -------------------------------------------------------------------------
    # 4. Extract Brands
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. BRAND EXTRACTION")
    print("=" * 80)

    print("\n--- Parsing CATEGORIES for brands ---")
    brands = []
    for cat_str in tqdm(df['CATEGORIES'], desc="  Parsing"):
        parsed = parse_categories(cat_str)
        brands.append(parsed['brand'])
    df['brand'] = brands

    brand_counts = df['brand'].value_counts()
    top_brands = set(brand_counts.head(TOP_BRAND_N).index)
    print(f"  Unique brands: {brand_counts.dropna().shape[0]}")
    print(f"  Has brand: {df['brand'].notna().sum()} ({df['brand'].notna().mean()*100:.2f}%)")
    print(f"  Top {TOP_BRAND_N} brands selected")

    # -------------------------------------------------------------------------
    # 5. Feature Preparation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. FEATURE PREPARATION")
    print("=" * 80)

    # --- Sparse Features ---
    print("\n--- Sparse Features (Embeddings) ---")

    # Position: 1-64 -> 0-63, unknown -> 64
    df['position_idx'] = df['position'].clip(1, 64) - 1
    n_positions = 65
    print(f"  position_idx: 0-{df['position_idx'].max()} (n_positions={n_positions})")

    # Placement: encode as integers
    placement_encoder = LabelEncoder()
    df['placement_idx'] = placement_encoder.fit_transform(df['placement'].astype(str))
    n_placements = len(placement_encoder.classes_) + 1
    print(f"  placement_idx: 0-{df['placement_idx'].max()} (n_placements={n_placements})")
    print(f"    Placement mapping: {dict(zip(placement_encoder.classes_, range(len(placement_encoder.classes_))))}")

    # Brand: top N -> 0 to N-1, other -> N, unknown -> N+1
    brand_to_idx = {brand: idx for idx, brand in enumerate(brand_counts.head(TOP_BRAND_N).index)}
    def get_brand_idx(brand):
        if pd.isna(brand):
            return TOP_BRAND_N + 1  # unknown
        return brand_to_idx.get(brand, TOP_BRAND_N)  # other

    df['brand_idx'] = df['brand'].apply(get_brand_idx)
    n_brands = TOP_BRAND_N + 2
    print(f"  brand_idx: 0-{df['brand_idx'].max()} (n_brands={n_brands})")
    print(f"    Top brands: {len(brand_to_idx)}, Other: {(df['brand_idx'] == TOP_BRAND_N).sum()}, Unknown: {(df['brand_idx'] == TOP_BRAND_N + 1).sum()}")

    # --- Dense Features ---
    print("\n--- Dense Features (Normalized) ---")

    dense_feature_names = ['quality', 'log_quality', 'bid', 'log_bid',
                           'log_price', 'relative_price_in_session',
                           'n_items', 'log_n_items']

    df['log_quality'] = np.log1p(df['quality'])
    df['log_bid'] = np.log1p(df['bid'])
    df['log_price'] = np.log1p(df['CATALOG_PRICE'])
    df['log_n_items'] = np.log1p(df['n_items'])

    # Relative price in session
    session_avg_price = df.groupby('auction_id')['CATALOG_PRICE'].transform('mean')
    df['relative_price_in_session'] = df['CATALOG_PRICE'] / (session_avg_price + 1)

    dense_features_df = df[dense_feature_names].copy()
    print(f"  Dense features ({len(dense_feature_names)}): {dense_feature_names}")

    for col in dense_feature_names:
        print(f"    {col}: mean={dense_features_df[col].mean():.4f}, std={dense_features_df[col].std():.4f}")

    # --- Product Embeddings ---
    print("\n--- Product Embeddings ---")

    product_embed_dim = product_embeddings_all.shape[1]
    print(f"  Embedding dimension: {product_embed_dim}")

    # Map product_id to embedding
    product_embeddings = np.zeros((len(df), product_embed_dim), dtype=np.float32)
    missing_embeddings = 0
    for i, pid in enumerate(tqdm(df['product_id'], desc="  Mapping embeddings")):
        idx = product_id_to_idx.get(pid, -1)
        if idx >= 0 and idx < len(product_embeddings_all):
            product_embeddings[i] = product_embeddings_all[idx]
        else:
            missing_embeddings += 1

    print(f"  Missing embeddings: {missing_embeddings} ({missing_embeddings/len(df)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 6. Train/Val/Test Split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    # Split by auction_id to avoid leakage
    auction_ids = df['auction_id'].unique()
    train_auctions, temp_auctions = train_test_split(
        auction_ids, test_size=0.3, random_state=RANDOM_STATE
    )
    val_auctions, test_auctions = train_test_split(
        temp_auctions, test_size=0.5, random_state=RANDOM_STATE
    )

    train_mask = df['auction_id'].isin(train_auctions)
    val_mask = df['auction_id'].isin(val_auctions)
    test_mask = df['auction_id'].isin(test_auctions)

    print(f"\n  Train auctions: {len(train_auctions)}")
    print(f"  Val auctions: {len(val_auctions)}")
    print(f"  Test auctions: {len(test_auctions)}")
    print(f"  Train samples: {train_mask.sum()} (CTR: {df.loc[train_mask, 'clicked'].mean():.4f})")
    print(f"  Val samples: {val_mask.sum()} (CTR: {df.loc[val_mask, 'clicked'].mean():.4f})")
    print(f"  Test samples: {test_mask.sum()} (CTR: {df.loc[test_mask, 'clicked'].mean():.4f})")

    # Prepare arrays
    sparse_cols = ['position_idx', 'placement_idx', 'brand_idx']

    X_sparse_train = df.loc[train_mask, sparse_cols].values
    X_sparse_val = df.loc[val_mask, sparse_cols].values
    X_sparse_test = df.loc[test_mask, sparse_cols].values

    X_dense_train = dense_features_df.loc[train_mask].values
    X_dense_val = dense_features_df.loc[val_mask].values
    X_dense_test = dense_features_df.loc[test_mask].values

    X_product_train = product_embeddings[train_mask.values]
    X_product_val = product_embeddings[val_mask.values]
    X_product_test = product_embeddings[test_mask.values]

    y_train = df.loc[train_mask, 'clicked'].values.astype(np.float32)
    y_val = df.loc[val_mask, 'clicked'].values.astype(np.float32)
    y_test = df.loc[test_mask, 'clicked'].values.astype(np.float32)

    # Normalize dense features
    scaler = StandardScaler()
    X_dense_train = scaler.fit_transform(X_dense_train).astype(np.float32)
    X_dense_val = scaler.transform(X_dense_val).astype(np.float32)
    X_dense_test = scaler.transform(X_dense_test).astype(np.float32)

    print("\n  Normalization stats (mean/std after fit on train):")
    for i, col in enumerate(dense_feature_names):
        print(f"    {col}: mean={scaler.mean_[i]:.4f}, scale={scaler.scale_[i]:.4f}")

    # Create datasets
    train_dataset = CTRDataset(X_sparse_train, X_dense_train, X_product_train, y_train)
    val_dataset = CTRDataset(X_sparse_val, X_dense_val, X_product_val, y_val)
    test_dataset = CTRDataset(X_sparse_test, X_dense_test, X_product_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # -------------------------------------------------------------------------
    # 7. Model Architecture
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. MODEL ARCHITECTURE")
    print("=" * 80)

    model = DeepFMCTR(
        n_positions=n_positions,
        n_placements=n_placements,
        n_brands=n_brands,
        product_embed_dim=product_embed_dim,
        n_dense_features=len(dense_feature_names),
        embed_dim_position=EMBED_DIM_POSITION,
        embed_dim_placement=EMBED_DIM_PLACEMENT,
        embed_dim_brand=EMBED_DIM_BRAND,
        mlp_dims=MLP_DIMS,
        dropout=DROPOUT,
        use_fm=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model: DeepFM-style CTR")
    print(f"  Device: {device}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n  Architecture:")
    print(f"    Sparse Embeddings:")
    print(f"      - position: Embedding({n_positions}, {EMBED_DIM_POSITION}) = {n_positions * EMBED_DIM_POSITION:,} params")
    print(f"      - placement: Embedding({n_placements}, {EMBED_DIM_PLACEMENT}) = {n_placements * EMBED_DIM_PLACEMENT:,} params")
    print(f"      - brand: Embedding({n_brands}, {EMBED_DIM_BRAND}) = {n_brands * EMBED_DIM_BRAND:,} params")
    print(f"      - product: Pre-computed({product_embed_dim})")

    total_embed_dim = EMBED_DIM_POSITION + EMBED_DIM_PLACEMENT + EMBED_DIM_BRAND + product_embed_dim
    total_input_dim = total_embed_dim + len(dense_feature_names)
    print(f"    Total embedding dim: {total_embed_dim}")
    print(f"    Total input dim (embed + dense): {total_input_dim}")

    print(f"    FM Layer: FMLayer({total_input_dim}, 10)")
    print(f"    Deep Layers (MLP):")
    input_dim = total_input_dim
    for i, hidden_dim in enumerate(MLP_DIMS):
        print(f"      - Linear({input_dim}, {hidden_dim}) + ReLU + Dropout({DROPOUT})")
        input_dim = hidden_dim
    print(f"      - Linear({input_dim}, 1)")
    print(f"    Output: Sigmoid(FM_out + Deep_out + bias)")

    print(f"\n  Model summary:")
    print(model)

    # -------------------------------------------------------------------------
    # 8. Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. TRAINING")
    print("=" * 80)

    print(f"\n  Config:")
    print(f"    Epochs: {EPOCHS}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Learning rate: {LEARNING_RATE}")
    print(f"    Weight decay: {WEIGHT_DECAY}")
    print(f"    Early stopping patience: {EARLY_STOP_PATIENCE}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    history = []

    print("\n  Training progress:")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val AUC':>10} | {'Status':>10}")
    print("  " + "-" * 60)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        })

        status = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            status = "BEST"
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                status = "STOP"

        print(f"  {epoch+1:>5} | {train_loss:>10.6f} | {val_loss:>10.6f} | {val_auc:>10.4f} | {status:>10}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  Loaded best model with Val AUC: {best_val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 9. Test Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("9. TEST EVALUATION")
    print("=" * 80)

    test_loss, test_auc, y_pred_test, y_true_test = evaluate(model, test_loader, criterion, device)

    # Additional metrics
    test_logloss = log_loss(y_true_test, y_pred_test)
    precision, recall, _ = precision_recall_curve(y_true_test, y_pred_test)
    test_pr_auc = auc(recall, precision)

    # Calibration
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_test, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred = y_pred_test[mask].mean()
            mean_actual = y_true_test[mask].mean()
            calibration_data.append({
                'bin': i,
                'pred_range': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                'n_samples': mask.sum(),
                'mean_pred': mean_pred,
                'mean_actual': mean_actual,
                'diff': mean_pred - mean_actual
            })

    print(f"\n  Test Metrics:")
    print(f"    AUC-ROC: {test_auc:.4f}")
    print(f"    Log Loss: {test_logloss:.4f}")
    print(f"    PR-AUC: {test_pr_auc:.4f}")
    print(f"    BCE Loss: {test_loss:.6f}")

    print(f"\n  Calibration (predicted vs actual by decile):")
    print(f"    {'Bin':>5} | {'Pred Range':>12} | {'N':>8} | {'Mean Pred':>10} | {'Mean Actual':>10} | {'Diff':>8}")
    print("    " + "-" * 65)
    for row in calibration_data:
        print(f"    {row['bin']:>5} | {row['pred_range']:>12} | {row['n_samples']:>8} | {row['mean_pred']:>10.4f} | {row['mean_actual']:>10.4f} | {row['diff']:>+8.4f}")

    # -------------------------------------------------------------------------
    # 10. Comparison with Baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. COMPARISON WITH BASELINE")
    print("=" * 80)

    baseline_auc = 0.6007
    baseline_logloss = 0.1214  # approximate from v17
    baseline_pr_auc = 0.0764   # approximate from v17

    print("\n  Metric            Baseline (v17)    DeepFM (v18)    Improvement")
    print("  " + "-" * 65)
    print(f"  AUC-ROC           {baseline_auc:.4f}            {test_auc:.4f}           {(test_auc - baseline_auc)*100:+.2f}%")
    print(f"  Log Loss          {baseline_logloss:.4f}            {test_logloss:.4f}           {(baseline_logloss - test_logloss)*100:+.2f}%")
    print(f"  PR-AUC            {baseline_pr_auc:.4f}            {test_pr_auc:.4f}           {(test_pr_auc - baseline_pr_auc)*100:+.2f}%")

    print(f"\n  Baseline: XGBoost with TF-IDF (112 features)")
    print(f"  DeepFM: Learned embeddings + FM + MLP ({total_params:,} parameters)")

    # -------------------------------------------------------------------------
    # 11. Embedding Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("11. EMBEDDING ANALYSIS")
    print("=" * 80)

    # Position embeddings
    print("\n--- Position Embeddings ---")
    pos_embed_weights = model.embed_position.weight.detach().cpu().numpy()
    pos_embed_norms = np.linalg.norm(pos_embed_weights, axis=1)

    print(f"  Embedding matrix shape: {pos_embed_weights.shape}")
    print(f"  Embedding norm by position (first 20):")
    for i in range(min(20, len(pos_embed_norms))):
        print(f"    Position {i+1}: norm={pos_embed_norms[i]:.4f}")

    # Brand embeddings
    print("\n--- Brand Embeddings (Top 20 by norm) ---")
    brand_embed_weights = model.embed_brand.weight.detach().cpu().numpy()
    brand_embed_norms = np.linalg.norm(brand_embed_weights, axis=1)

    # Map back to brand names
    idx_to_brand = {v: k for k, v in brand_to_idx.items()}
    brand_norm_data = []
    for idx, norm in enumerate(brand_embed_norms):
        if idx in idx_to_brand:
            brand_norm_data.append({'brand': idx_to_brand[idx], 'idx': idx, 'norm': norm})
        elif idx == TOP_BRAND_N:
            brand_norm_data.append({'brand': '[OTHER]', 'idx': idx, 'norm': norm})
        elif idx == TOP_BRAND_N + 1:
            brand_norm_data.append({'brand': '[UNKNOWN]', 'idx': idx, 'norm': norm})

    brand_norm_df = pd.DataFrame(brand_norm_data).sort_values('norm', ascending=False)
    print(f"  Embedding matrix shape: {brand_embed_weights.shape}")
    print(f"  Top 20 brands by embedding norm:")
    for i, row in brand_norm_df.head(20).iterrows():
        print(f"    {row['brand'][:30]:30s}: norm={row['norm']:.4f}")

    # Placement embeddings
    print("\n--- Placement Embeddings ---")
    place_embed_weights = model.embed_placement.weight.detach().cpu().numpy()
    place_embed_norms = np.linalg.norm(place_embed_weights, axis=1)

    for idx, cls in enumerate(placement_encoder.classes_):
        print(f"  Placement {cls}: norm={place_embed_norms[idx]:.4f}")

    # -------------------------------------------------------------------------
    # 12. Feature Importance via Gradient
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("12. FEATURE IMPORTANCE (Gradient-based)")
    print("=" * 80)

    # Get gradients for a batch
    model.train()
    sample_batch = next(iter(test_loader))
    sparse, dense, product_emb, labels = [x.to(device) for x in sample_batch]

    sparse.requires_grad = False
    dense.requires_grad = True
    product_emb.requires_grad = True

    outputs = model(sparse, dense, product_emb)
    loss = criterion(outputs, labels)
    loss.backward()

    # Dense feature importance
    dense_grad = dense.grad.abs().mean(dim=0).cpu().numpy()
    print("\n--- Dense Feature Importance (mean absolute gradient) ---")
    for i, (name, grad) in enumerate(zip(dense_feature_names, dense_grad)):
        print(f"  {name:30s}: {grad:.6f}")

    # Product embedding importance
    product_grad = product_emb.grad.abs().mean().item()
    print(f"\n--- Product Embedding Importance ---")
    print(f"  Mean absolute gradient: {product_grad:.6f}")

    # -------------------------------------------------------------------------
    # 13. Prediction Distribution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("13. PREDICTION DISTRIBUTION")
    print("=" * 80)

    print(f"\n  Test set prediction statistics:")
    print(f"    Min: {y_pred_test.min():.6f}")
    print(f"    Max: {y_pred_test.max():.6f}")
    print(f"    Mean: {y_pred_test.mean():.6f}")
    print(f"    Std: {y_pred_test.std():.6f}")
    print(f"    Median: {np.median(y_pred_test):.6f}")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}th: {np.percentile(y_pred_test, p):.6f}")

    # By position
    print(f"\n  Mean prediction by position (first 20):")
    df_test = df[test_mask].copy()
    df_test['pred'] = y_pred_test
    pos_preds = df_test.groupby('position')['pred'].mean().head(20)
    pos_actuals = df_test.groupby('position')['clicked'].mean().head(20)
    for pos in pos_preds.index:
        print(f"    Position {int(pos):2d}: pred={pos_preds[pos]:.4f}, actual={pos_actuals[pos]:.4f}, diff={pos_preds[pos]-pos_actuals[pos]:+.4f}")

    # -------------------------------------------------------------------------
    # 14. Training History
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("14. TRAINING HISTORY")
    print("=" * 80)

    print(f"\n  Full training history:")
    print(f"  {'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val AUC':>10}")
    print("  " + "-" * 50)
    for h in history:
        print(f"  {h['epoch']:>5} | {h['train_loss']:>12.6f} | {h['val_loss']:>12.6f} | {h['val_auc']:>10.4f}")

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
  - Train/Val/Test split: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}

Model Architecture:
  - Type: DeepFM (Factorization Machine + Deep Neural Network)
  - Sparse features: position, placement, brand (learned embeddings)
  - Dense features: quality, bid, price, session context ({len(dense_feature_names)} features)
  - Product embeddings: Pre-computed ({product_embed_dim}-dim)
  - FM layer: Learns pairwise interactions
  - MLP: {MLP_DIMS}
  - Total parameters: {total_params:,}
  - Device: {device}

Training:
  - Epochs trained: {len(history)} (early stopped at patience={EARLY_STOP_PATIENCE})
  - Best validation AUC: {best_val_auc:.4f}
  - Final train loss: {history[-1]['train_loss']:.6f}
  - Final val loss: {history[-1]['val_loss']:.6f}

Test Performance:
  - AUC-ROC: {test_auc:.4f}
  - Log Loss: {test_logloss:.4f}
  - PR-AUC: {test_pr_auc:.4f}

Comparison with Baseline (v17 XGBoost + TF-IDF):
  - Baseline AUC: {baseline_auc:.4f}
  - DeepFM AUC: {test_auc:.4f}
  - Improvement: {(test_auc - baseline_auc)*100:+.2f}%
""")

    print("=" * 80)
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 80)

    output.close()
    sys.stdout = output.stdout


if __name__ == '__main__':
    main()
