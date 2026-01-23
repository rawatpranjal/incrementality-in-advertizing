"""
Standalone synthetic data generator for FPPE simulation.
Loads the fitted copula model and generates synthetic auction data.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import pickle
import argparse

def generate_synthetic(copula_model, n_samples=10000, seed=None):
    """
    Generate synthetic samples from Gaussian copula.

    Parameters:
    -----------
    copula_model : dict
        Fitted copula model with correlation matrix and marginals
    n_samples : int
        Number of synthetic samples to generate
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic data with same structure as original
    """

    if seed is not None:
        np.random.seed(seed)

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


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic auction data for FPPE')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='synthetic_data.parquet',
                        help='Output file path')
    parser.add_argument('--model', type=str, default='simple_copula_model.pkl',
                        help='Path to copula model file')

    args = parser.parse_args()

    print("="*80)
    print("SYNTHETIC DATA GENERATOR FOR FPPE")
    print("="*80)
    print()

    # Load copula model
    print(f"Loading copula model from {args.model}...")
    with open(args.model, 'rb') as f:
        copula_model = pickle.load(f)

    print(f"  Variables: {copula_model['variables']}")
    print(f"  Training sample size: {copula_model['n_samples']:,}")
    print()

    # Generate synthetic data
    print(f"Generating {args.n_samples:,} synthetic samples (seed={args.seed})...")
    synthetic = generate_synthetic(copula_model, n_samples=args.n_samples, seed=args.seed)
    print("✓ Synthetic data generated")
    print()

    # Summary
    print("Synthetic Data Summary:")
    print(synthetic.describe())
    print()

    # Save
    print(f"Saving to {args.output}...")
    synthetic.to_parquet(args.output)
    print(f"✓ Saved {len(synthetic):,} rows")
    print()

    # Key statistics for FPPE
    print("="*80)
    print("KEY PARAMETERS FOR FPPE SIMULATION")
    print("="*80)
    print()

    print("BUDGETS (assuming daily aggregation):")
    print(f"  Median: ${synthetic['value'].median():.4f}")
    print(f"  Range (p10-p90): ${synthetic['value'].quantile(0.10):.4f} - ${synthetic['value'].quantile(0.90):.4f}")
    print()

    print("VALUATIONS:")
    print(f"  Median: ${synthetic['value'].median():.4f}")
    print(f"  Range (p10-p90): ${synthetic['value'].quantile(0.10):.4f} - ${synthetic['value'].quantile(0.90):.4f}")
    print()

    print("BIDS:")
    print(f"  Median: ${synthetic['FINAL_BID_DOLLARS'].median():.4f}")
    print(f"  Range (p10-p90): ${synthetic['FINAL_BID_DOLLARS'].quantile(0.10):.4f} - ${synthetic['FINAL_BID_DOLLARS'].quantile(0.90):.4f}")
    print()

    print("PACING:")
    print(f"  Mean: {synthetic['PACING'].mean():.4f}")
    print(f"  % at pacing >= 0.95: {(synthetic['PACING'] >= 0.95).mean()*100:.1f}%")
    print()

    print("CONVERSION RATE:")
    print(f"  Median: {synthetic['CONVERSION_RATE'].median():.6f}")
    print(f"  Range (p10-p90): {synthetic['CONVERSION_RATE'].quantile(0.10):.6f} - {synthetic['CONVERSION_RATE'].quantile(0.90):.6f}")
    print()

    print("QUALITY:")
    print(f"  Median: {synthetic['QUALITY'].median():.6f}")
    print(f"  Range (p10-p90): {synthetic['QUALITY'].quantile(0.10):.6f} - {synthetic['QUALITY'].quantile(0.90):.6f}")
    print()


if __name__ == '__main__':
    main()
