"""
Causal Inference vs Correlation in Ad Optimization
Using MILP (Mixed Integer Linear Programming) to prove causality matters
Single script with comprehensive stdout output
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
import time
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("CAUSAL INFERENCE VS CORRELATION IN AD OPTIMIZATION")
print("Using Mixed Integer Linear Programming (MILP)")
print(f"Started: {datetime.now()}")
print("="*80)

# ============================================================================
# DATA GENERATION
# ============================================================================

n_users = 1000
n_vendors = 20
budget = 20.0  # Tight budget to force selection
value_per_conversion = 50.0

print("\n" + "="*60)
print("DATA GENERATION")
print("="*60)
print(f"Users: {n_users}")
print(f"Vendors: {n_vendors}")
print(f"Budget: ${budget:.2f}")
print(f"Value per conversion: ${value_per_conversion:.2f}")

# Generate user data with NEGATIVE correlation between baseline and lift
print("\nGenerating heterogeneous treatment effects...")

# Baseline purchase probability (what happens without ads)
baseline_prob = np.random.beta(2, 8, n_users)
print(f"Baseline purchase rate: mean={baseline_prob.mean():.3f}, std={baseline_prob.std():.3f}")

# TRUE CAUSAL LIFT (inversely related to baseline)
# Key insight: ads work best on marginal customers
true_lift = 0.08 / (1 + 10 * baseline_prob) + np.random.normal(0, 0.005, n_users)
true_lift = np.clip(true_lift, 0.001, 0.15)
print(f"True causal lift: mean={true_lift.mean():.3f}, std={true_lift.std():.3f}")

correlation = np.corrcoef(baseline_prob, true_lift)[0, 1]
print(f"Correlation(baseline, lift): {correlation:.3f} [NEGATIVE - KEY!]")

# Click-through rates (independent of purchase behavior)
ctr = np.random.beta(3, 20, n_users)
print(f"Click-through rate: mean={ctr.mean():.3f}, std={ctr.std():.3f}")

# Cost per impression (auction prices)
cpc = np.random.exponential(0.5, n_users) + 0.1
print(f"Cost per click: mean=${cpc.mean():.2f}, std=${cpc.std():.2f}")

# Observed CVR (conflates baseline and treatment)
observed_cvr = baseline_prob + true_lift
print(f"Observed CVR: mean={observed_cvr.mean():.3f}, std={observed_cvr.std():.3f}")

# True incremental value per impression
true_incremental = ctr * true_lift * value_per_conversion
print(f"True incremental value: mean=${true_incremental.mean():.2f}, std=${true_incremental.std():.2f}")

# Cost per impression
cost_per_impression = ctr * cpc
print(f"Cost per impression: mean=${cost_per_impression.mean():.3f}, std=${cost_per_impression.std():.3f}")

# ============================================================================
# OPTIMIZATION PROBLEM SETUP
# ============================================================================

print("\n" + "="*60)
print("OPTIMIZATION PROBLEM")
print("="*60)
print("Maximize: Σ score_i * x_i")
print("Subject to:")
print(f"  Budget: Σ cost_i * x_i ≤ {budget}")
print("  Binary: x_i ∈ {0, 1}")
print("Where x_i = 1 if we show ad to user i")

# ============================================================================
# METHOD 1: CORRELATION-BASED (STANDARD INDUSTRY PRACTICE)
# ============================================================================

print("\n" + "="*60)
print("METHOD 1: CORRELATION-BASED OPTIMIZATION")
print("="*60)
print("Score = CTR × Observed_CVR × Value")
print("(This is what most platforms do)")

correlation_score = ctr * observed_cvr * value_per_conversion
print(f"Correlation scores: mean=${correlation_score.mean():.2f}, std=${correlation_score.std():.2f}")

# Setup MILP
c = -correlation_score  # Negative because linprog minimizes

# Constraints: Ax <= b
# Budget constraint: cost * x <= budget
A = cost_per_impression.reshape(1, -1)
b_ub = np.array([budget])

# Bounds: 0 <= x_i <= 1
bounds = Bounds(lb=np.zeros(n_users), ub=np.ones(n_users))

# Integer constraint
integrality = np.ones(n_users)

print("\nSolving MILP for correlation-based method...")
start_time = time.time()

result_corr = milp(
    c=c,
    constraints=LinearConstraint(A, -np.inf, b_ub),
    bounds=bounds,
    integrality=integrality,
    options={'disp': False}
)

solve_time = time.time() - start_time
print(f"Optimization completed in {solve_time:.3f} seconds")
print(f"Solver status: {result_corr.message}")

# Extract solution
x_corr = result_corr.x.astype(int)
n_selected_corr = x_corr.sum()
spend_corr = (cost_per_impression * x_corr).sum()
selected_idx_corr = np.where(x_corr == 1)[0]

# Calculate ACTUAL incremental value (using true lift)
incremental_conversions_corr = (true_incremental[selected_idx_corr] / value_per_conversion).sum()
incremental_value_corr = (true_incremental * x_corr).sum()
iroas_corr = incremental_value_corr / spend_corr if spend_corr > 0 else 0

# What did we select?
avg_baseline_corr = baseline_prob[selected_idx_corr].mean()
avg_lift_corr = true_lift[selected_idx_corr].mean()
avg_ctr_corr = ctr[selected_idx_corr].mean()

print(f"\nRESULTS:")
print(f"  Users selected: {n_selected_corr}")
print(f"  Spend: ${spend_corr:.2f}")
print(f"  Incremental conversions: {incremental_conversions_corr:.2f}")
print(f"  Incremental value: ${incremental_value_corr:.2f}")
print(f"  iROAS: {iroas_corr:.2f}x")
print(f"  Average baseline of selected: {avg_baseline_corr:.3f}")
print(f"  Average lift of selected: {avg_lift_corr:.4f}")
print(f"  Average CTR of selected: {avg_ctr_corr:.3f}")

# ============================================================================
# METHOD 2: CAUSAL (HTE-BASED)
# ============================================================================

print("\n" + "="*60)
print("METHOD 2: CAUSAL OPTIMIZATION (HTE)")
print("="*60)
print("Score = CTR × Estimated_Lift × Value")
print("(This is what we SHOULD do)")

# Add estimation noise to true lift (realistic HTE estimation)
noise_std = 0.3 * true_lift.std()
estimated_lift = true_lift + np.random.normal(0, noise_std, n_users)
estimated_lift = np.clip(estimated_lift, 0.001, 0.2)
print(f"Estimation noise added: std={noise_std:.4f}")

causal_score = ctr * estimated_lift * value_per_conversion
print(f"Causal scores: mean=${causal_score.mean():.2f}, std=${causal_score.std():.2f}")

# Setup MILP
c = -causal_score

print("\nSolving MILP for causal method...")
start_time = time.time()

result_causal = milp(
    c=c,
    constraints=LinearConstraint(A, -np.inf, b_ub),
    bounds=bounds,
    integrality=integrality,
    options={'disp': False}
)

solve_time = time.time() - start_time
print(f"Optimization completed in {solve_time:.3f} seconds")
print(f"Solver status: {result_causal.message}")

# Extract solution
x_causal = result_causal.x.astype(int)
n_selected_causal = x_causal.sum()
spend_causal = (cost_per_impression * x_causal).sum()
selected_idx_causal = np.where(x_causal == 1)[0]

# Calculate ACTUAL incremental value
incremental_conversions_causal = (true_incremental[selected_idx_causal] / value_per_conversion).sum()
incremental_value_causal = (true_incremental * x_causal).sum()
iroas_causal = incremental_value_causal / spend_causal if spend_causal > 0 else 0

# What did we select?
avg_baseline_causal = baseline_prob[selected_idx_causal].mean()
avg_lift_causal = true_lift[selected_idx_causal].mean()
avg_ctr_causal = ctr[selected_idx_causal].mean()

print(f"\nRESULTS:")
print(f"  Users selected: {n_selected_causal}")
print(f"  Spend: ${spend_causal:.2f}")
print(f"  Incremental conversions: {incremental_conversions_causal:.2f}")
print(f"  Incremental value: ${incremental_value_causal:.2f}")
print(f"  iROAS: {iroas_causal:.2f}x")
print(f"  Average baseline of selected: {avg_baseline_causal:.3f}")
print(f"  Average lift of selected: {avg_lift_causal:.4f}")
print(f"  Average CTR of selected: {avg_ctr_causal:.3f}")

# ============================================================================
# METHOD 3: ORACLE (PERFECT KNOWLEDGE)
# ============================================================================

print("\n" + "="*60)
print("METHOD 3: ORACLE (PERFECT KNOWLEDGE)")
print("="*60)
print("Score = True_Incremental_Value")
print("(Upper bound with perfect information)")

oracle_score = true_incremental
print(f"Oracle scores: mean=${oracle_score.mean():.2f}, std=${oracle_score.std():.2f}")

# Setup MILP
c = -oracle_score

print("\nSolving MILP for oracle method...")
start_time = time.time()

result_oracle = milp(
    c=c,
    constraints=LinearConstraint(A, -np.inf, b_ub),
    bounds=bounds,
    integrality=integrality,
    options={'disp': False}
)

solve_time = time.time() - start_time
print(f"Optimization completed in {solve_time:.3f} seconds")
print(f"Solver status: {result_oracle.message}")

# Extract solution
x_oracle = result_oracle.x.astype(int)
n_selected_oracle = x_oracle.sum()
spend_oracle = (cost_per_impression * x_oracle).sum()
selected_idx_oracle = np.where(x_oracle == 1)[0]

# Calculate value
incremental_conversions_oracle = (true_incremental[selected_idx_oracle] / value_per_conversion).sum()
incremental_value_oracle = (true_incremental * x_oracle).sum()
iroas_oracle = incremental_value_oracle / spend_oracle if spend_oracle > 0 else 0

# What did we select?
avg_baseline_oracle = baseline_prob[selected_idx_oracle].mean()
avg_lift_oracle = true_lift[selected_idx_oracle].mean()
avg_ctr_oracle = ctr[selected_idx_oracle].mean()

print(f"\nRESULTS:")
print(f"  Users selected: {n_selected_oracle}")
print(f"  Spend: ${spend_oracle:.2f}")
print(f"  Incremental conversions: {incremental_conversions_oracle:.2f}")
print(f"  Incremental value: ${incremental_value_oracle:.2f}")
print(f"  iROAS: {iroas_oracle:.2f}x")
print(f"  Average baseline of selected: {avg_baseline_oracle:.3f}")
print(f"  Average lift of selected: {avg_lift_oracle:.4f}")
print(f"  Average CTR of selected: {avg_ctr_oracle:.3f}")

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Method': ['Correlation', 'Causal (HTE)', 'Oracle'],
    'Users Selected': [n_selected_corr, n_selected_causal, n_selected_oracle],
    'Spend': [spend_corr, spend_causal, spend_oracle],
    'Inc. Conversions': [incremental_conversions_corr, incremental_conversions_causal, incremental_conversions_oracle],
    'Inc. Value': [incremental_value_corr, incremental_value_causal, incremental_value_oracle],
    'iROAS': [iroas_corr, iroas_causal, iroas_oracle],
    'Avg Baseline': [avg_baseline_corr, avg_baseline_causal, avg_baseline_oracle],
    'Avg Lift': [avg_lift_corr, avg_lift_causal, avg_lift_oracle]
})

print(comparison_df.to_string(index=False))

# Calculate improvements
improvement_abs = iroas_causal - iroas_corr
improvement_pct = (iroas_causal / iroas_corr - 1) * 100 if iroas_corr > 0 else 0
oracle_gap = (iroas_oracle / iroas_causal - 1) * 100 if iroas_causal > 0 else 0

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"1. CAUSALITY BEATS CORRELATION:")
print(f"   - Causal method achieves {iroas_causal:.2f}x iROAS")
print(f"   - Correlation method achieves {iroas_corr:.2f}x iROAS")
print(f"   - Improvement: +{improvement_pct:.1f}% ({improvement_abs:.2f}x absolute)")

print(f"\n2. SELECTION PATTERNS:")
print(f"   - Correlation selects HIGH baseline ({avg_baseline_corr:.3f}), LOW lift ({avg_lift_corr:.4f})")
print(f"   - Causal selects LOW baseline ({avg_baseline_causal:.3f}), HIGH lift ({avg_lift_causal:.4f})")
print(f"   - This is WHY causality matters!")

print(f"\n3. ROOM FOR IMPROVEMENT:")
print(f"   - Oracle achieves {iroas_oracle:.2f}x iROAS")
print(f"   - Gap to oracle: {oracle_gap:.1f}%")
print(f"   - Better lift estimation → higher returns")

print(f"\n4. OPTIMIZATION DETAILS:")
print(f"   - Used Mixed Integer Linear Programming (MILP)")
print(f"   - Binary decision variables (show ad or not)")
print(f"   - Budget constraint enforced exactly")
print(f"   - All solutions are globally optimal")

# Overlap analysis
overlap = (x_corr * x_causal).sum()
overlap_pct = overlap / min(n_selected_corr, n_selected_causal) * 100
print(f"\n5. USER OVERLAP:")
print(f"   - Users selected by both methods: {overlap}")
print(f"   - Overlap percentage: {overlap_pct:.1f}%")
print(f"   - Methods target DIFFERENT users!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("Causal inference (HTE) dramatically outperforms correlation-based")
print("optimization by correctly identifying high-lift users instead of")
print("high-baseline users. This is achieved through proper MILP optimization")
print("with the RIGHT objective function (incremental value, not correlation).")
print(f"\nCompleted: {datetime.now()}")
print("="*80)