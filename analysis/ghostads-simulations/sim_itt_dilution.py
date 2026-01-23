"""
Simulation: ITT Dilution

Verifies Appendix A.2: τ_ITT = π · τ_ATT

Theory:
-------
Under random assignment, one-sided noncompliance, and SUTVA:
    τ_ITT = E[Y_i | Z_i=1] - E[Y_i | Z_i=0] = π · τ_ATT

where π = P(D_i=1 | Z_i=1) is the compliance rate and
τ_ATT = E[Y_i(1) - Y_i(0) | D_i=1] is the average treatment effect on the treated.

Simulation Design:
------------------
- Sample size N = 2000
- Vary compliance rate π ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- True treatment effect τ = 2.0 (constant across all units)
- Monte Carlo iterations: 10,000 per π
- For each iteration, calculate empirical τ_ITT and compare to π · τ_ATT

Expected Results:
-----------------
1. Empirical τ_ITT should equal π · τ for all π
2. As π decreases, τ_ITT should decrease proportionally
3. The ratio τ_ITT / τ_ATT should equal π
"""

import numpy as np
from tqdm import tqdm
import sys
from common_dgp import GhostAdsDGP

def run_itt_dilution_simulation():
    """Run simulation to verify ITT dilution."""

    # Simulation parameters
    N = 2000
    tau = 2.0
    sigma_0 = 2.0
    sigma_1 = 2.0
    n_sims = 10000
    seed = 42

    # Vary compliance rate
    pi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Results storage
    results = []

    print("=" * 80)
    print("ITT DILUTION SIMULATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sample size N: {N}")
    print(f"  True treatment effect τ: {tau}")
    print(f"  Control outcome SD σ_0: {sigma_0}")
    print(f"  Treatment outcome SD σ_1: {sigma_1}")
    print(f"  Monte Carlo iterations: {n_sims}")
    print(f"  Random seed: {seed}")
    print(f"\nCompliance rates π: {pi_values}")
    print("\n" + "=" * 80)

    for pi in pi_values:
        print(f"\n\nRunning simulation for π = {pi}...")

        # Storage for this π
        itt_estimates = []
        late_estimates = []
        first_stages = []

        # Monte Carlo loop
        for i in tqdm(range(n_sims), desc=f"π={pi}", file=sys.stdout):
            dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0,
                              sigma_1=None, seed=seed + i)

            # Estimate ITT
            itt_res = dgp.estimate_itt()
            itt_estimates.append(itt_res['estimate'])

            # Estimate LATE (should equal ATT under one-sided noncompliance)
            late_res = dgp.estimate_late()
            late_estimates.append(late_res['estimate'])
            first_stages.append(late_res['first_stage'])

        # Convert to arrays
        itt_estimates = np.array(itt_estimates)
        late_estimates = np.array(late_estimates)
        first_stages = np.array(first_stages)

        # Remove NaN values (from zero first stage)
        valid = ~np.isnan(late_estimates)
        late_estimates_valid = late_estimates[valid]

        # Calculate statistics
        mean_itt = np.mean(itt_estimates)
        sd_itt = np.std(itt_estimates, ddof=1)
        mean_late = np.mean(late_estimates_valid)
        sd_late = np.std(late_estimates_valid, ddof=1)
        mean_first_stage = np.mean(first_stages)

        # Theoretical predictions
        tau_itt_theory = pi * tau
        tau_att_theory = tau  # ATT equals tau since effect is constant

        # Dilution ratio
        dilution_ratio_empirical = mean_itt / tau
        dilution_ratio_theory = pi

        # Store results
        results.append({
            'pi': pi,
            'mean_itt': mean_itt,
            'sd_itt': sd_itt,
            'mean_late': mean_late,
            'sd_late': sd_late,
            'mean_first_stage': mean_first_stage,
            'tau_itt_theory': tau_itt_theory,
            'tau_att_theory': tau_att_theory,
            'dilution_ratio_empirical': dilution_ratio_empirical,
            'dilution_ratio_theory': dilution_ratio_theory,
            'bias_itt': mean_itt - tau_itt_theory,
            'bias_late': mean_late - tau_att_theory,
            'n_valid_late': len(late_estimates_valid)
        })

        print(f"\n  Results for π = {pi}:")
        print(f"    Empirical ITT: {mean_itt:.6f} (SD: {sd_itt:.6f})")
        print(f"    Theoretical ITT: {tau_itt_theory:.6f}")
        print(f"    Bias: {mean_itt - tau_itt_theory:.6f}")
        print(f"\n    Empirical LATE/ATT: {mean_late:.6f} (SD: {sd_late:.6f})")
        print(f"    Theoretical ATT: {tau_att_theory:.6f}")
        print(f"    Bias: {mean_late - tau_att_theory:.6f}")
        print(f"\n    Mean first stage: {mean_first_stage:.6f}")
        print(f"    Dilution ratio (empirical): {dilution_ratio_empirical:.6f}")
        print(f"    Dilution ratio (theory): {dilution_ratio_theory:.6f}")
        print(f"    Ratio difference: {abs(dilution_ratio_empirical - dilution_ratio_theory):.6f}")

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE: ITT DILUTION")
    print("=" * 80)
    print(f"\n{'π':<8} {'ITT (Emp)':<12} {'ITT (Theory)':<12} {'Bias':<12} {'Dilution Ratio':<16} {'Match?':<8}")
    print("-" * 80)

    for r in results:
        match = "✓" if abs(r['dilution_ratio_empirical'] - r['dilution_ratio_theory']) < 0.01 else "✗"
        print(f"{r['pi']:<8.2f} {r['mean_itt']:<12.6f} {r['tau_itt_theory']:<12.6f} "
              f"{r['bias_itt']:<12.6f} {r['dilution_ratio_empirical']:<16.6f} {match:<8}")

    # Verification
    print("\n\n" + "=" * 80)
    print("VERIFICATION: τ_ITT = π · τ_ATT")
    print("=" * 80)

    all_match = True
    for r in results:
        match = abs(r['dilution_ratio_empirical'] - r['dilution_ratio_theory']) < 0.01
        all_match = all_match and match

    if all_match:
        print("\n✓ SUCCESS: All dilution ratios match theory within tolerance (< 0.01)")
        print("✓ The simulation confirms: τ_ITT = π · τ_ATT")
    else:
        print("\n✗ FAILURE: Some dilution ratios do not match theory")
        print("✗ Check the simulation setup or DGP")

    # Check linear relationship
    print("\n\nChecking linearity: τ_ITT = π · τ")
    pi_array = np.array([r['pi'] for r in results])
    itt_array = np.array([r['mean_itt'] for r in results])

    # OLS regression of ITT on π
    slope = np.sum((pi_array - np.mean(pi_array)) * (itt_array - np.mean(itt_array))) / \
            np.sum((pi_array - np.mean(pi_array))**2)
    intercept = np.mean(itt_array) - slope * np.mean(pi_array)
    r_squared = np.corrcoef(pi_array, itt_array)[0, 1]**2

    print(f"\n  Regression: τ_ITT = {intercept:.6f} + {slope:.6f} · π")
    print(f"  R² = {r_squared:.6f}")
    print(f"\n  Expected slope: {tau:.6f}")
    print(f"  Empirical slope: {slope:.6f}")
    print(f"  Difference: {abs(slope - tau):.6f}")

    if abs(slope - tau) < 0.05 and r_squared > 0.99:
        print("\n✓ Linear relationship confirmed: ITT = π · τ")
    else:
        print("\n✗ Linear relationship not perfectly confirmed")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_itt_dilution_simulation()
