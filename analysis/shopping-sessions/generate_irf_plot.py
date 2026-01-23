"""
Generate Impulse Response Function (IRF) plots for click ad stock effects.

This script:
1. Loads the estimated model_full from the Netflix panel analysis
2. Extracts click ad stock coefficients and variance-covariance matrix
3. Calculates the instantaneous and cumulative IRF over 72 hours
4. Generates publication-quality plots with confidence intervals
5. Saves output to latex directory for inclusion in paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumulative_trapezoid
from pyfixest.estimation import feols
import warnings
warnings.filterwarnings('ignore')

# Load panel data
print("Loading Netflix panel data...")
panel = pd.read_parquet('data/netflix_panel.parquet')
panel = panel[panel['sample_type'] != 'double_negative'].copy()

# Calculate vendor-level aggregates (needed for model specification)
vendor_stats = panel.groupby('vendor_id').agg({
    'adstock_imp_1day': 'mean',
    'outcome': 'sum'
}).rename(columns={
    'adstock_imp_1day': 'vendor_avg_adstock',
    'outcome': 'vendor_conversions'
})
panel = panel.merge(vendor_stats, on='vendor_id', how='left')

print(f"Panel loaded: {len(panel):,} observations")

# Re-estimate the full model to get coefficients and vcov
print("\nRe-estimating full model...")
formula_full = (
    "outcome ~ adstock_imp_1hr + adstock_imp_3hr + adstock_imp_1day + "
    "adstock_click_1hr + adstock_click_3hr + adstock_click_1day + "
    "auction_stock_6hr + click_stock_1hr + "
    "impressions_7d + clicks_7d + "
    "vendor_avg_adstock + "
    "sin_hour + cos_hour + sin_dow + cos_dow + is_weekend + is_evening | "
    "user_id + week_id"
)

model_full = feols(
    fml=formula_full,
    data=panel,
    weights='sample_weight',
    vcov='hetero'
)

print("Model estimation complete.")
# Note: pyfixest Feols objects don't have r2_within attribute directly accessible

# Extract coefficients and covariance matrix
coefs = model_full.coef()

# Get the variance-covariance matrix from the internal _vcov attribute
# This is a numpy array with shape (n_coefs, n_coefs)
vcov_matrix = model_full._vcov

# Convert to DataFrame for easier indexing
coef_names = coefs.index.tolist()
vcov_df = pd.DataFrame(vcov_matrix, index=coef_names, columns=coef_names)

# Define time grid for plotting (72 hours)
time_grid_hours = np.linspace(0.01, 72, 200)

# --- CLICK IMPULSE RESPONSE FUNCTION ---
print("\nCalculating click IRF...")

# Get relevant click coefficients
click_vars = ['adstock_click_1hr', 'adstock_click_3hr', 'adstock_click_1day']
beta_click = coefs[click_vars].values
vcov_click = vcov_df.loc[click_vars, click_vars].values

print(f"\nClick coefficients:")
for var, coef in zip(click_vars, beta_click):
    print(f"  {var}: {coef:.4f}")

# Define decay rates (half-lives in hours)
h1, h2, h3 = 1, 3, 24
d1, d2, d3 = np.log(2) / h1, np.log(2) / h2, np.log(2) / h3

# Create the ad stock values over time for a single click at t=0
X_click = np.column_stack([
    np.exp(-d1 * time_grid_hours),  # 1hr decay
    np.exp(-d2 * time_grid_hours),  # 3hr decay
    np.exp(-d3 * time_grid_hours)   # 1day decay
])

# Calculate point estimate of IRF: E[y(t) | click at t=0]
irf_click = X_click @ beta_click

# Calculate standard error: se(X'β) = sqrt(X' * Var(β) * X)
se_irf_click = np.sqrt(np.diag(X_click @ vcov_click @ X_click.T))

# Calculate 95% confidence intervals
ci_lower_click = irf_click - 1.96 * se_irf_click
ci_upper_click = irf_click + 1.96 * se_irf_click

# --- CUMULATIVE IRF (Integral over time) ---
print("\nCalculating cumulative IRF...")

# Integrate the instantaneous effect to get cumulative effect
cumulative_irf_click = cumulative_trapezoid(irf_click, time_grid_hours, initial=0)

# For confidence intervals, integrate the bounds
# Note: This is a conservative approximation; proper uncertainty propagation would require bootstrap
cumulative_ci_lower = cumulative_trapezoid(ci_lower_click, time_grid_hours, initial=0)
cumulative_ci_upper = cumulative_trapezoid(ci_upper_click, time_grid_hours, initial=0)

print(f"\nTotal cumulative effect at 72 hours: {cumulative_irf_click[-1]:.6f}")
print(f"95% CI: [{cumulative_ci_lower[-1]:.6f}, {cumulative_ci_upper[-1]:.6f}]")

# --- PLOTTING ---
print("\nGenerating plots...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Instantaneous Effect (IRF)
axes[0].plot(time_grid_hours, irf_click, label='Estimated Effect', color='navy', linewidth=2)
axes[0].fill_between(time_grid_hours, ci_lower_click, ci_upper_click,
                      color='cornflowerblue', alpha=0.3, label='95% Confidence Interval')
axes[0].axhline(0, color='grey', linestyle='--', linewidth=1)
axes[0].set_title('Instantaneous Causal Effect of a Click on Conversion Probability', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hours Since Click', fontsize=12)
axes[0].set_ylabel('Change in Conversion Probability', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].set_xlim(0, 72)
axes[0].grid(alpha=0.3)

# Plot 2: Cumulative Effect
axes[1].plot(time_grid_hours, cumulative_irf_click, label='Cumulative Effect', color='darkgreen', linewidth=2)
axes[1].fill_between(time_grid_hours, cumulative_ci_lower, cumulative_ci_upper,
                      color='lightgreen', alpha=0.3, label='95% Confidence Interval')
axes[1].axhline(0, color='grey', linestyle='--', linewidth=1)
axes[1].set_title('Cumulative Causal Effect of a Click on Conversions', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hours Since Click', fontsize=12)
axes[1].set_ylabel('Total Incremental Conversions per Click', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 72)
axes[1].grid(alpha=0.3)

plt.tight_layout()

# Save to latex directory
output_path = '../latex/click_impulse_response.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"\nPlot saved to: {output_path}")

# Also save as PNG for preview
plt.savefig('../latex/click_impulse_response.png', bbox_inches='tight', dpi=300)
print(f"PNG preview saved to: ../latex/click_impulse_response.png")

plt.show()

print("\n" + "="*80)
print("IRF GENERATION COMPLETE")
print("="*80)
print("\nKey findings:")
print(f"1. Peak effect occurs at ~{time_grid_hours[np.argmax(irf_click)]:.1f} hours")
print(f"2. Effect becomes statistically insignificant after ~{time_grid_hours[np.where(ci_lower_click > 0)[0][-1] if any(ci_lower_click > 0) else 0]:.1f} hours")
print(f"3. Total incremental conversions per click: {cumulative_irf_click[-1]:.4f}")
print(f"4. 95% of total effect captured within first {time_grid_hours[np.where(cumulative_irf_click >= 0.95*cumulative_irf_click[-1])[0][0] if cumulative_irf_click[-1] > 0 else 0]:.1f} hours")