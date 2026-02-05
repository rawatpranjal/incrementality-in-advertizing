"""
Simulation: Precision Comparison

Verifies Appendix A.11: Precision ranking across estimators

Theory:
-------
From Appendix A.11, the precision comparison states:
    Var(τ̂_LATE) ≈ (1/π) × Var(τ̂_PSA)
                 ≈ (1/π) × Var(θ̂_GA)
                 ≈ (1/π) × Var(τ̂_PGA)

This means:
1. PSA, Ghost Ads, and PGA are more precise than LATE by factor 1/π
2. PSA and Ghost Ads have identical variance
3. PGA has the same asymptotic scaling as GA, with small penalty when p̂ < 1

Simulation Design:
------------------
- Sample size N = 2000
- Vary compliance rate π ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- True treatment effect τ = 2.0
- Monte Carlo iterations: 10,000
- Calculate empirical MSE, variance, and relative efficiency

Expected Results:
-----------------
1. MSE(LATE) / MSE(GA) ≈ 1/π
2. MSE(PSA) ≈ MSE(GA) ≈ MSE(PGA)
3. As π decreases, LATE becomes increasingly inefficient relative to GA
4. PSA, GA, PGA maintain consistent relative efficiency regardless of π
"""

import numpy as np
from tqdm import tqdm
import sys
from common_dgp import GhostAdsDGP

def run_precision_comparison_simulation():
    """Run simulation to compare precision across estimators."""

    # Simulation parameters
    N = 2000
    tau = 2.0
    sigma_0 = 2.0
    sigma_1 = 2.0
    p_predict = 0.97
    n_sims = 10000
    seed = 42

    # Vary compliance rate
    pi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("=" * 80)
    print("PRECISION COMPARISON SIMULATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sample size N: {N}")
    print(f"  True treatment effect τ: {tau}")
    print(f"  Control outcome SD σ_0: {sigma_0}")
    print(f"  Treatment outcome SD σ_1: {sigma_1}")
    print(f"  PGA prediction accuracy: {p_predict}")
    print(f"  Monte Carlo iterations: {n_sims}")
    print(f"  Random seed: {seed}")
    print(f"\nCompliance rates π: {pi_values}")
    print("\n" + "=" * 80)

    # Results storage
    results = []

    for pi in pi_values:
        print(f"\n\nRunning simulation for π = {pi}...")

        # Storage for this π
        late_estimates = []
        psa_estimates = []
        ga_estimates = []
        pga_estimates = []

        # Monte Carlo loop
        for i in tqdm(range(n_sims), desc=f"π={pi}", file=sys.stdout):
            dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0,
                              sigma_1=None, p_predict=p_predict, seed=seed + i)

            # Estimate all estimators
            late_res = dgp.estimate_late()
            late_estimates.append(late_res['estimate'])

            psa_res = dgp.estimate_psa()
            psa_estimates.append(psa_res['estimate'])

            ga_res = dgp.estimate_ghost_ads()
            ga_estimates.append(ga_res['estimate'])

            pga_res = dgp.estimate_pga()
            pga_estimates.append(pga_res['estimate'])

        # Convert to arrays and remove NaNs
        late_estimates = np.array(late_estimates)
        psa_estimates = np.array(psa_estimates)
        ga_estimates = np.array(ga_estimates)
        pga_estimates = np.array(pga_estimates)

        late_valid = late_estimates[~np.isnan(late_estimates)]
        psa_valid = psa_estimates[~np.isnan(psa_estimates)]
        ga_valid = ga_estimates[~np.isnan(ga_estimates)]
        pga_valid = pga_estimates[~np.isnan(pga_estimates)]

        # True effects
        tau_att = tau  # ATT = τ since effect is constant

        # Calculate statistics
        stats_late = {
            'mean': np.mean(late_valid),
            'variance': np.var(late_valid, ddof=1),
            'sd': np.std(late_valid, ddof=1),
            'bias': np.mean(late_valid) - tau_att,
            'mse': np.mean((late_valid - tau_att)**2),
            'rmse': np.sqrt(np.mean((late_valid - tau_att)**2)),
            'n_valid': len(late_valid)
        }

        stats_psa = {
            'mean': np.mean(psa_valid),
            'variance': np.var(psa_valid, ddof=1),
            'sd': np.std(psa_valid, ddof=1),
            'bias': np.mean(psa_valid) - tau_att,
            'mse': np.mean((psa_valid - tau_att)**2),
            'rmse': np.sqrt(np.mean((psa_valid - tau_att)**2)),
            'n_valid': len(psa_valid)
        }

        stats_ga = {
            'mean': np.mean(ga_valid),
            'variance': np.var(ga_valid, ddof=1),
            'sd': np.std(ga_valid, ddof=1),
            'bias': np.mean(ga_valid) - tau_att,
            'mse': np.mean((ga_valid - tau_att)**2),
            'rmse': np.sqrt(np.mean((ga_valid - tau_att)**2)),
            'n_valid': len(ga_valid)
        }

        stats_pga = {
            'mean': np.mean(pga_valid),
            'variance': np.var(pga_valid, ddof=1),
            'sd': np.std(pga_valid, ddof=1),
            'bias': np.mean(pga_valid) - tau_att,
            'mse': np.mean((pga_valid - tau_att)**2),
            'rmse': np.sqrt(np.mean((pga_valid - tau_att)**2)),
            'n_valid': len(pga_valid)
        }

        # Relative efficiency (using Ghost Ads as baseline)
        rel_eff_late = stats_ga['mse'] / stats_late['mse']
        rel_eff_psa = stats_ga['mse'] / stats_psa['mse']
        rel_eff_pga = stats_ga['mse'] / stats_pga['mse']

        # Variance ratios
        var_ratio_late_ga = stats_late['variance'] / stats_ga['variance']
        var_ratio_psa_ga = stats_psa['variance'] / stats_ga['variance']
        var_ratio_pga_ga = stats_pga['variance'] / stats_ga['variance']

        # Store results
        results.append({
            'pi': pi,
            'stats_late': stats_late,
            'stats_psa': stats_psa,
            'stats_ga': stats_ga,
            'stats_pga': stats_pga,
            'rel_eff_late': rel_eff_late,
            'rel_eff_psa': rel_eff_psa,
            'rel_eff_pga': rel_eff_pga,
            'var_ratio_late_ga': var_ratio_late_ga,
            'var_ratio_psa_ga': var_ratio_psa_ga,
            'var_ratio_pga_ga': var_ratio_pga_ga
        })

        print(f"\n  Results for π = {pi}:")
        print(f"    LATE: SD = {stats_late['sd']:.6f}, MSE = {stats_late['mse']:.6f}")
        print(f"    PSA:  SD = {stats_psa['sd']:.6f}, MSE = {stats_psa['mse']:.6f}")
        print(f"    GA:   SD = {stats_ga['sd']:.6f}, MSE = {stats_ga['mse']:.6f}")
        print(f"    PGA:  SD = {stats_pga['sd']:.6f}, MSE = {stats_pga['mse']:.6f}")
        print(f"\n    Variance ratio Var(LATE)/Var(GA): {var_ratio_late_ga:.4f} (theory: {1/pi:.4f})")

    # Summary tables
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE 1: VARIANCE COMPARISON")
    print("=" * 80)
    print(f"\n{'π':<8} {'SD(LATE)':<12} {'SD(PSA)':<12} {'SD(GA)':<12} {'SD(PGA)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['pi']:<8.2f} {r['stats_late']['sd']:<12.6f} {r['stats_psa']['sd']:<12.6f} "
              f"{r['stats_ga']['sd']:<12.6f} {r['stats_pga']['sd']:<12.6f}")

    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE 2: VARIANCE RATIOS")
    print("=" * 80)
    print(f"\n{'π':<8} {'Var(LATE)/Var(GA)':<20} {'Theory (1/π)':<15} {'Match?':<10} {'Var(PSA)/Var(GA)':<20}")
    print("-" * 80)
    for r in results:
        theory = 1 / r['pi']
        match = "✓" if abs(r['var_ratio_late_ga'] - theory) / theory < 0.15 else "✗"
        print(f"{r['pi']:<8.2f} {r['var_ratio_late_ga']:<20.6f} {theory:<15.6f} {match:<10} "
              f"{r['var_ratio_psa_ga']:<20.6f}")

    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE 3: RELATIVE EFFICIENCY (vs Ghost Ads)")
    print("=" * 80)
    print(f"\n{'π':<8} {'LATE':<15} {'PSA':<15} {'PGA':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['pi']:<8.2f} {r['rel_eff_late']:<15.6f} {r['rel_eff_psa']:<15.6f} "
              f"{r['rel_eff_pga']:<15.6f}")

    print("\nNote: Relative efficiency = MSE(GA) / MSE(estimator)")
    print("      Values < 1 mean Ghost Ads is more efficient")
    print("      Values ≈ 1 mean similar efficiency")

    # Verification
    print("\n\n" + "=" * 80)
    print("VERIFICATION: PRECISION COMPARISON")
    print("=" * 80)

    # Check 1: Var(LATE) / Var(GA) ≈ 1/π
    print("\n1. Checking Var(LATE) / Var(GA) ≈ 1/π:")
    variance_checks = []
    for r in results:
        theory = 1 / r['pi']
        empirical = r['var_ratio_late_ga']
        match = abs(empirical - theory) / theory < 0.15
        variance_checks.append(match)
        status = "✓" if match else "✗"
        print(f"  {status} π = {r['pi']:.2f}: empirical = {empirical:.4f}, theory = {theory:.4f}, "
              f"diff = {abs(empirical - theory):.4f}")

    # Check 2: PSA, GA, PGA have similar variance
    print("\n2. Checking PSA, GA, PGA have similar variance:")
    consistency_checks = []
    for r in results:
        var_psa = r['stats_psa']['variance']
        var_ga = r['stats_ga']['variance']
        var_pga = r['stats_pga']['variance']

        # Check if all three are within 20% of each other
        vars = [var_psa, var_ga, var_pga]
        mean_var = np.mean(vars)
        max_diff_pct = max([abs(v - mean_var) / mean_var for v in vars])

        match = max_diff_pct < 0.20
        consistency_checks.append(match)
        status = "✓" if match else "✗"
        print(f"  {status} π = {r['pi']:.2f}: PSA = {var_psa:.4f}, GA = {var_ga:.4f}, "
              f"PGA = {var_pga:.4f}, max diff = {max_diff_pct*100:.2f}%")

    # Check 3: LATE becomes less efficient as π decreases
    print("\n3. Checking LATE becomes less efficient as π decreases:")
    rel_effs = [r['rel_eff_late'] for r in results]
    pis = [r['pi'] for r in results]

    # Check if relative efficiency decreases with π
    # (i.e., correlation between π and rel_eff should be positive)
    correlation = np.corrcoef(pis, rel_effs)[0, 1]
    trend_check = correlation > 0.9

    status = "✓" if trend_check else "✗"
    print(f"  {status} Correlation between π and relative efficiency: {correlation:.4f}")
    print(f"      (Should be > 0.9, meaning LATE is less efficient at lower π)")

    # Check 4: PSA, GA, PGA maintain consistent efficiency across π
    print("\n4. Checking PSA, GA, PGA maintain consistent efficiency across π:")
    psa_effs = [r['rel_eff_psa'] for r in results]
    pga_effs = [r['rel_eff_pga'] for r in results]

    psa_stable = np.std(psa_effs) < 0.05
    pga_stable = np.std(pga_effs) < 0.05

    status_psa = "✓" if psa_stable else "✗"
    status_pga = "✓" if pga_stable else "✗"
    print(f"  {status_psa} PSA relative efficiency SD: {np.std(psa_effs):.6f} (should be < 0.05)")
    print(f"  {status_pga} PGA relative efficiency SD: {np.std(pga_effs):.6f} (should be < 0.05)")

    # Final verdict
    all_checks = variance_checks + consistency_checks + [trend_check, psa_stable, pga_stable]

    print("\n" + "=" * 80)
    if all(all_checks):
        print("✓ SUCCESS: All precision comparisons confirmed")
        print("\nKey findings:")
        print("  ✓ Var(LATE) / Var(GA) ≈ 1/π for all π")
        print("  ✓ PSA, Ghost Ads, and PGA have similar variance")
        print("  ✓ LATE becomes less efficient as π decreases")
        print("  ✓ PSA, GA, PGA maintain consistent efficiency regardless of π")
        print("\nConclusion:")
        print("  PSA, Ghost Ads, and PGA are more precise than LATE by factor 1/π")
        print("  Ghost Ads achieves PSA-level precision without placebo contamination")
        print("  PGA achieves Ghost Ads-level precision with imperfect prediction")
    else:
        print("✗ PARTIAL SUCCESS: Some precision comparisons not perfectly confirmed")
        print(f"  {sum(all_checks)}/{len(all_checks)} checks passed")

    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_precision_comparison_simulation()
