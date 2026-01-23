# Ghost Ads: Proof by Simulation

This directory contains Monte Carlo simulations that verify the theoretical results from the Ghost Ads paper appendix.

## Overview

The simulations use a common data generating process (DGP) based on the potential outcomes framework to verify:

1. **ITT Dilution** (Appendix A.2): τ_ITT = π · τ_ATT
2. **Variance Scaling Laws** (Appendix A.3-A.10):
   - LATE: Var(τ̂) ∝ 1/(Nπ²)
   - PSA/Ghost Ads/PGA: Var(τ̂) ∝ 1/(Nπ)
3. **Unbiased Identification** (Appendix A.2-A.9): All estimators are unbiased under their respective assumptions
4. **Precision Comparison** (Appendix A.11): Var(τ̂_LATE) ≈ (1/π) × Var(τ̂_PSA) ≈ (1/π) × Var(τ̂_GA) ≈ (1/π) × Var(τ̂_PGA)

## Files

- **common_dgp.py**: Shared data generating process with potential outcomes framework
- **sim_itt_dilution.py**: Verifies ITT dilution for various compliance rates π
- **sim_identification.py**: Verifies unbiased estimation for all estimators
- **sim_variance_scaling.py**: Verifies variance scaling laws across sample sizes and π
- **sim_precision_comparison.py**: Compares empirical variance ratios across estimators
- **results/**: Output directory containing verbose .txt files

## Data Generating Process

The common DGP implements:

1. Potential outcomes Y_i(0), Y_i(1) drawn from normal distributions with configurable:
   - True treatment effect τ (default: 2.0)
   - Control outcome variance σ²_0 (default: 4.0)
   - Treatment outcome variance σ²_1 (default: 4.0)

2. Random assignment Z_i ∈ {0,1} with 50/50 split

3. Actual exposure D_i with compliance rate π:
   - One-sided noncompliance: D_i = 0 for all Z_i = 0
   - In treatment: D_i(1) ~ Bernoulli(π)

4. Ghost ad indicators GA_i for control users:
   - Counterfactual tagging: GA_i = 1 ⟺ D_i(T) = 1
   - Simulates perfect auction simulator

5. Predicted exposure D̂_i with prediction accuracy p:
   - Pre-randomization shadow auction simulation
   - D̂_i ⊥ Z_i (independence assumption)

6. Placebo exposure indicators for PSA:
   - Perfect blind: D_i(CP) = P ⟺ D_i(T) = 1
   - No placebo effect: Y_i(P) = Y_i(0)

## Estimators

Each simulation implements:

1. **ITT**: τ̂_ITT = Ȳ_1 - Ȳ_0
2. **LATE**: τ̂_LATE = (Ȳ_1 - Ȳ_0) / (D̄_1 - D̄_0)
3. **PSA**: τ̂_PSA = Ȳ_{T,D=1} - Ȳ_{CP,D=P}
4. **Ghost Ads**: θ̂_GA = Ȳ_{T,D=1} - Ȳ_{C,GA=1}
5. **PGA**: τ̂_PGA = (Ȳ_{T,D̂=1} - Ȳ_{C,D̂=1}) / p̂

## Parameters

Default simulation parameters:
- Sample sizes N: [500, 1000, 2000, 5000]
- Compliance rates π: [0.1, 0.3, 0.5, 0.7, 0.9]
- True treatment effect τ: 2.0
- Monte Carlo iterations: 10,000 per configuration
- Random seed: 42 (for reproducibility)

## Output Format

Each simulation produces one .txt file in results/ containing:
- Configuration parameters
- Empirical statistics: mean, bias, variance, MSE, coverage
- Theoretical predictions
- Comparison ratios (empirical vs theoretical)
- All output is verbose with full precision (no rounding)

## Running Simulations

Run individual simulations:
```bash
python sim_itt_dilution.py
python sim_identification.py
python sim_variance_scaling.py
python sim_precision_comparison.py
```

Each script is self-contained and produces one results file.
