#!/usr/bin/env python3
"""
EDA Q7: How deterministic is the Auction Rank?

Regress RANKING on FINAL_BID, QUALITY, and PACING. What is the R²?
If it's low, the auction has significant unobserved noise (good for identification).
If it's 1.0, the system is deterministic.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent  # staggered-adoption/
EDA_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = EDA_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "07_auction_rank_determinism.txt"

# =============================================================================
# LOGGING
# =============================================================================
def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# =============================================================================
# SNOWFLAKE CONNECTION
# =============================================================================
def get_snowflake_connection():
    """Establish Snowflake connection using environment variables."""
    try:
        import snowflake.connector
        load_dotenv()

        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            database='INCREMENTALITY',
            schema='INCREMENTALITY_RESEARCH'
        )
        return conn
    except Exception as e:
        return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        log("=" * 80, f)
        log("EDA Q7: AUCTION RANK DETERMINISM", f)
        log("=" * 80, f)
        log("", f)

        log("QUESTION:", f)
        log("  How deterministic is the auction ranking system?", f)
        log("  Low R² = Noise in rankings (good for identification).", f)
        log("  R²=1 = Fully deterministic (may limit variation).", f)
        log("", f)

        log("METHODOLOGY:", f)
        log("  1. Regress RANKING ~ FINAL_BID + QUALITY + PACING", f)
        log("  2. Add interaction term (FINAL_BID × QUALITY)", f)
        log("  3. Report R² and coefficient significance", f)
        log("", f)
        log("=" * 80, f)
        log("", f)

        # -----------------------------------------------------------------
        # Connect to Snowflake
        # -----------------------------------------------------------------
        log("CONNECTING TO SNOWFLAKE...", f)

        conn = get_snowflake_connection()

        if conn is None:
            log("  [ERROR] Could not connect to Snowflake.", f)
            log("  QUALITY, PACING columns are not in local sample data.", f)
            log("  This analysis requires Snowflake access.", f)
            log("", f)
            analyze_local_sample(f)
            return

        log("  [SUCCESS] Snowflake connection established.", f)
        log("", f)

        # -----------------------------------------------------------------
        # Query auction data with ranking factors
        # -----------------------------------------------------------------
        log("QUERYING AUCTION RANKING DATA...", f)
        log("-" * 40, f)

        query = """
        SELECT
            RANKING,
            FINAL_BID,
            QUALITY,
            PACING,
            CONVERSION_RATE,
            IS_WINNER
        FROM AUCTIONS_RESULTS
        WHERE CREATED_AT BETWEEN '2025-09-01' AND '2025-09-07'
          AND FINAL_BID IS NOT NULL
          AND QUALITY IS NOT NULL
          AND PACING IS NOT NULL
          AND RANKING IS NOT NULL
        LIMIT 500000
        """

        try:
            with tqdm(desc="Executing query") as pbar:
                df = pd.read_sql(query, conn)
                pbar.update(1)

            log(f"  [SUCCESS] Query completed. {len(df):,} rows.", f)
            log("", f)

            conn.close()

        except Exception as e:
            log(f"  [ERROR] Query failed: {e}", f)
            return

        # -----------------------------------------------------------------
        # Descriptive statistics
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("DESCRIPTIVE STATISTICS", f)
        log("-" * 40, f)
        log("", f)

        for col in ['RANKING', 'FINAL_BID', 'QUALITY', 'PACING', 'CONVERSION_RATE']:
            if col in df.columns:
                stats = df[col].describe()
                log(f"  {col}:", f)
                log(f"    Mean: {stats['mean']:.4f}", f)
                log(f"    Std:  {stats['std']:.4f}", f)
                log(f"    Min:  {stats['min']:.4f}", f)
                log(f"    Max:  {stats['max']:.4f}", f)
                log("", f)

        # -----------------------------------------------------------------
        # Correlation matrix
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("CORRELATION MATRIX", f)
        log("-" * 40, f)
        log("", f)

        numeric_cols = ['RANKING', 'FINAL_BID', 'QUALITY', 'PACING']
        if 'CONVERSION_RATE' in df.columns and df['CONVERSION_RATE'].notna().any():
            numeric_cols.append('CONVERSION_RATE')

        corr_matrix = df[numeric_cols].corr()

        log(f"  {'':<18}", end="", f=f)
        for col in numeric_cols:
            log(f"{col[:12]:>14}", end="", f=f)
        log("", f)

        for row in numeric_cols:
            log(f"  {row:<18}", end="", f=f)
            for col in numeric_cols:
                log(f"{corr_matrix.loc[row, col]:>14.3f}", end="", f=f)
            log("", f)

        log("", f)

        # -----------------------------------------------------------------
        # OLS Regression: RANKING ~ FINAL_BID + QUALITY + PACING
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("OLS REGRESSION: RANKING ~ FINAL_BID + QUALITY + PACING", f)
        log("-" * 40, f)
        log("", f)

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        # Prepare data
        X_cols = ['FINAL_BID', 'QUALITY', 'PACING']
        X = df[X_cols].dropna()
        y = df.loc[X.index, 'RANKING']

        log(f"  Sample size: {len(X):,}", f)
        log("", f)

        # Model 1: Simple linear
        model1 = LinearRegression()
        model1.fit(X, y)
        y_pred1 = model1.predict(X)
        r2_1 = r2_score(y, y_pred1)

        log("  MODEL 1: Linear (no interactions)", f)
        log(f"    R²: {r2_1:.4f}", f)
        log("", f)
        log("    Coefficients:", f)
        for i, col in enumerate(X_cols):
            log(f"      {col}: {model1.coef_[i]:.6f}", f)
        log(f"      Intercept: {model1.intercept_:.6f}", f)
        log("", f)

        # Model 2: With interaction term
        X_interact = X.copy()
        X_interact['BID_x_QUALITY'] = X['FINAL_BID'] * X['QUALITY']

        model2 = LinearRegression()
        model2.fit(X_interact, y)
        y_pred2 = model2.predict(X_interact)
        r2_2 = r2_score(y, y_pred2)

        log("  MODEL 2: With interaction (FINAL_BID × QUALITY)", f)
        log(f"    R²: {r2_2:.4f}", f)
        log("", f)
        log("    Coefficients:", f)
        for i, col in enumerate(X_interact.columns):
            log(f"      {col}: {model2.coef_[i]:.6f}", f)
        log(f"      Intercept: {model2.intercept_:.6f}", f)
        log("", f)

        # Model 3: Polynomial features
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        model3 = LinearRegression()
        model3.fit(X_poly, y)
        y_pred3 = model3.predict(X_poly)
        r2_3 = r2_score(y, y_pred3)

        log("  MODEL 3: Polynomial (degree=2)", f)
        log(f"    R²: {r2_3:.4f}", f)
        log(f"    Features: {X_poly.shape[1]}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Within-auction analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("WITHIN-AUCTION RANKING ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        # Check if ranking is purely determined by bid*quality
        df['bid_quality_score'] = df['FINAL_BID'] * df['QUALITY']

        # Rank correlation within auctions would need AUCTION_ID
        log("  RANK CORRELATION:", f)
        log(f"    Spearman(RANKING, FINAL_BID): {df['RANKING'].corr(df['FINAL_BID'], method='spearman'):.4f}", f)
        log(f"    Spearman(RANKING, QUALITY): {df['RANKING'].corr(df['QUALITY'], method='spearman'):.4f}", f)
        log(f"    Spearman(RANKING, BID×QUALITY): {df['RANKING'].corr(df['bid_quality_score'], method='spearman'):.4f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Residual analysis
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("RESIDUAL ANALYSIS", f)
        log("-" * 40, f)
        log("", f)

        residuals = y - y_pred2
        log(f"  Residual statistics (Model 2):", f)
        log(f"    Mean: {residuals.mean():.4f}", f)
        log(f"    Std:  {residuals.std():.4f}", f)
        log(f"    Min:  {residuals.min():.4f}", f)
        log(f"    Max:  {residuals.max():.4f}", f)
        log("", f)

        # Residual distribution
        log("  Residual distribution:", f)
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            val = np.percentile(residuals, pct)
            log(f"    P{pct:02d}: {val:.2f}", f)
        log("", f)

        # -----------------------------------------------------------------
        # Interpretation
        # -----------------------------------------------------------------
        log("=" * 80, f)
        log("INTERPRETATION", f)
        log("-" * 40, f)
        log("", f)

        log(f"  R² SUMMARY:", f)
        log(f"    Model 1 (linear):       {r2_1:.4f}", f)
        log(f"    Model 2 (interaction):  {r2_2:.4f}", f)
        log(f"    Model 3 (polynomial):   {r2_3:.4f}", f)
        log("", f)

        if r2_2 > 0.95:
            log("  [NOTE] Very high R² (>0.95).", f)
            log("  The auction ranking is highly deterministic.", f)
            log("  FINAL_BID, QUALITY, PACING explain most variance.", f)
            log("  Little room for unobserved factors.", f)
        elif r2_2 > 0.80:
            log("  [OK] High R² (0.80-0.95).", f)
            log("  Ranking is largely predictable from observables.", f)
            log("  Some residual variation exists.", f)
        elif r2_2 > 0.50:
            log("  [GOOD] Moderate R² (0.50-0.80).", f)
            log("  Significant unobserved variation in rankings.", f)
            log("  Good for instrumental variable strategies.", f)
        else:
            log("  [INTERESTING] Low R² (<0.50).", f)
            log("  Rankings have substantial unexplained variation.", f)
            log("  May indicate other factors (timing, context, etc.).", f)

        log("", f)
        log("  IMPLICATIONS FOR IDENTIFICATION:", f)
        log("    - If deterministic: Use discontinuity at rank thresholds", f)
        log("    - If noisy: Exploit randomness for natural experiments", f)
        log("", f)

        log("=" * 80, f)
        log("ANALYSIS COMPLETE", f)
        log("=" * 80, f)


def analyze_local_sample(f):
    """Analyze local sample when Snowflake unavailable."""
    log("", f)
    log("=" * 80, f)
    log("LOCAL SAMPLE RANKING ANALYSIS", f)
    log("-" * 40, f)
    log("", f)

    RAW_DATA_DIR = BASE_DIR / ".." / "shopping-sessions" / "data"
    ar_path = RAW_DATA_DIR / "raw_sample_auctions_results.parquet"

    if ar_path.exists():
        ar = pd.read_parquet(ar_path)
        log(f"  Sample auction results: {len(ar):,} rows", f)
        log(f"  Columns: {list(ar.columns)}", f)
        log("", f)

        if 'RANKING' in ar.columns:
            log("  RANKING column found.", f)
            log(f"    Min: {ar['RANKING'].min()}", f)
            log(f"    Max: {ar['RANKING'].max()}", f)
            log(f"    Mean: {ar['RANKING'].mean():.2f}", f)
            log("", f)

            # Analyze ranking vs winner status
            if 'IS_WINNER' in ar.columns:
                winner_ranks = ar[ar['IS_WINNER'] == True]['RANKING']
                loser_ranks = ar[ar['IS_WINNER'] == False]['RANKING']
                log("  Ranking by winner status:", f)
                log(f"    Winners - Mean rank: {winner_ranks.mean():.2f}", f)
                log(f"    Losers - Mean rank: {loser_ranks.mean():.2f}", f)
        else:
            log("  [NOTE] RANKING column available but QUALITY/PACING missing.", f)
            log("  Cannot perform full regression analysis.", f)
    else:
        log(f"  [ERROR] Sample file not found: {ar_path}", f)

    log("", f)


if __name__ == "__main__":
    main()
