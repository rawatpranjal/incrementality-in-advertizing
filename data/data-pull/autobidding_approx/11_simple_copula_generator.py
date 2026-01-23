"""
Simple Gaussian copula using scipy for fast synthetic data generation.
Captures correlation structure without complex marginal distribution fitting.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.stats import gaussian_kde
import pickle

print("="*80)
print("SIMPLE GAUSSIAN COPULA - PLACEMENT 5")
print("="*80)
print()

# Load Placement 5 data
print("Loading Placement 5 data...")
df = pd.read_parquet('placement5_data.parquet')
print(f"Total rows: {len(df):,}")
print()

# Select key variables for copula
variables = ['PACING', 'FINAL_BID_DOLLARS', 'CONVERSION_RATE', 'QUALITY', 'value']

print("Variables for copula modeling:")
for var in variables:
    print(f"  - {var}")
print()

# Prepare data: sample for efficiency
SAMPLE_SIZE = 50000
print(f"Sampling {SAMPLE_SIZE:,} rows...")
df_model = df[variables].dropna().sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
print(f"Sample size: {len(df_model):,}")
print()

print("="*80)
print("DATA SUMMARY")
print("="*80)
print()

print("Descriptive Statistics:")
print(df_model.describe())
print()

print("Pearson Correlation Matrix:")
corr_pearson = df_model.corr()
print(corr_pearson)
print()

print("Spearman Correlation Matrix:")
corr_spearman = df_model.corr(method='spearman')
print(corr_spearman)
print()

# Step 1: Transform to uniform [0,1] using empirical CDF (rank-based)
print("="*80)
print("STEP 1: TRANSFORM TO UNIFORM MARGINS")
print("="*80)
print()

U = pd.DataFrame(index=df_model.index)
for var in variables:
    # Empirical CDF: rank / (n+1)
    ranks = df_model[var].rank(method='average')
    U[var] = ranks / (len(df_model) + 1)
    print(f"{var}: min={U[var].min():.4f}, max={U[var].max():.4f}, mean={U[var].mean():.4f}")

print()

# Step 2: Transform to standard normal using inverse CDF
print("="*80)
print("STEP 2: TRANSFORM TO STANDARD NORMAL")
print("="*80)
print()

Z = pd.DataFrame(index=df_model.index)
for var in variables:
    Z[var] = norm.ppf(U[var])
    print(f"{var}: mean={Z[var].mean():.4f}, std={Z[var].std():.4f}")

print()

# Step 3: Estimate correlation matrix in normal space
print("="*80)
print("STEP 3: ESTIMATE CORRELATION MATRIX")
print("="*80)
print()

Sigma = Z.corr()
print("Correlation matrix (Sigma):")
print(Sigma)
print()

# Step 4: Save marginal distributions (empirical quantile functions)
print("="*80)
print("STEP 4: SAVE MARGINAL DISTRIBUTIONS")
print("="*80)
print()

marginals = {}
for var in variables:
    # Store sorted values for inverse transform (quantile function)
    marginals[var] = {
        'values': np.sort(df_model[var].values),
        'mean': df_model[var].mean(),
        'std': df_model[var].std(),
        'min': df_model[var].min(),
        'max': df_model[var].max(),
        'quantiles': {
            'p01': df_model[var].quantile(0.01),
            'p05': df_model[var].quantile(0.05),
            'p10': df_model[var].quantile(0.10),
            'p25': df_model[var].quantile(0.25),
            'p50': df_model[var].quantile(0.50),
            'p75': df_model[var].quantile(0.75),
            'p90': df_model[var].quantile(0.90),
            'p95': df_model[var].quantile(0.95),
            'p99': df_model[var].quantile(0.99),
        }
    }
    print(f"{var}:")
    print(f"  Range: [{marginals[var]['min']:.6f}, {marginals[var]['max']:.6f}]")
    print(f"  Mean: {marginals[var]['mean']:.6f}, Std: {marginals[var]['std']:.6f}")

print()

# Save copula model
copula_model = {
    'variables': variables,
    'correlation_matrix': Sigma,
    'marginals': marginals,
    'n_samples': len(df_model)
}

print("Saving copula model...")
with open('simple_copula_model.pkl', 'wb') as f:
    pickle.dump(copula_model, f)
print("✓ Saved to simple_copula_model.pkl")
print()

# Step 5: Generate synthetic samples
print("="*80)
print("STEP 5: GENERATE SYNTHETIC SAMPLES")
print("="*80)
print()

def generate_synthetic(copula_model, n_samples=10000):
    """Generate synthetic samples from Gaussian copula"""

    variables = copula_model['variables']
    Sigma = copula_model['correlation_matrix'].values
    marginals = copula_model['marginals']

    # Generate from multivariate normal
    Z_synthetic = np.random.multivariate_normal(
        mean=np.zeros(len(variables)),
        cov=Sigma,
        size=n_samples
    )

    # Transform to uniform
    U_synthetic = norm.cdf(Z_synthetic)

    # Transform to original marginals using empirical quantiles
    synthetic_data = pd.DataFrame()
    for i, var in enumerate(variables):
        u = U_synthetic[:, i]
        # Map uniform to original scale using empirical quantile function
        sorted_values = marginals[var]['values']
        indices = (u * len(sorted_values)).astype(int)
        indices = np.clip(indices, 0, len(sorted_values) - 1)
        synthetic_data[var] = sorted_values[indices]

    return synthetic_data

n_samples = 10000
print(f"Generating {n_samples:,} synthetic samples...")
synthetic = generate_synthetic(copula_model, n_samples)
print("✓ Synthetic data generated")
print()

print("Synthetic Data Summary:")
print(synthetic.describe())
print()

print("Synthetic Correlation Matrix:")
print(synthetic.corr())
print()

# Compare moments
print("="*80)
print("MOMENT COMPARISON (Real vs Synthetic)")
print("="*80)
print()

for var in variables:
    real_mean = df_model[var].mean()
    real_std = df_model[var].std()
    real_median = df_model[var].median()

    synth_mean = synthetic[var].mean()
    synth_std = synthetic[var].std()
    synth_median = synthetic[var].median()

    print(f"{var}:")
    print(f"  Mean:   Real={real_mean:.6f}, Synthetic={synth_mean:.6f}, Diff={(synth_mean-real_mean)/real_mean*100:.1f}%")
    print(f"  Std:    Real={real_std:.6f}, Synthetic={synth_std:.6f}, Diff={(synth_std-real_std)/real_std*100:.1f}%")
    print(f"  Median: Real={real_median:.6f}, Synthetic={synth_median:.6f}, Diff={(synth_median-real_median)/real_median*100:.1f}%")
    print()

# Compare correlations
print("="*80)
print("CORRELATION PRESERVATION")
print("="*80)
print()

corr_real = df_model.corr()
corr_synth = synthetic.corr()

print("Correlation Differences (Synthetic - Real):")
print(corr_synth - corr_real)
print()

# Calculate mean absolute correlation error
corr_error = np.abs(corr_synth.values - corr_real.values)
# Only off-diagonal elements
mask = ~np.eye(len(variables), dtype=bool)
mean_corr_error = corr_error[mask].mean()
print(f"Mean Absolute Correlation Error: {mean_corr_error:.6f}")
print()

if mean_corr_error < 0.05:
    print("✓ Excellent correlation preservation!")
elif mean_corr_error < 0.10:
    print("✓ Good correlation preservation")
else:
    print("⚠️  Moderate correlation preservation")

print()

# Save synthetic sample
print("Saving synthetic sample...")
synthetic.to_parquet('synthetic_sample_10k.parquet')
print("✓ Saved to synthetic_sample_10k.parquet")
print()

print("="*80)
print("USAGE EXAMPLE")
print("="*80)
print()
print("To generate more synthetic data:")
print()
print("  import pickle")
print("  with open('simple_copula_model.pkl', 'rb') as f:")
print("      copula_model = pickle.load(f)")
print()
print("  # Use the generate_synthetic function from this script")
print("  synthetic = generate_synthetic(copula_model, n_samples=1000)")
print()
