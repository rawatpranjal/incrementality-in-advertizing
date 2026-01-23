"""
Simulation: Variance Scaling Laws

Verifies Appendix A.11: Variance scaling across estimators

Theory:
-------
From the appendix, the variance scaling laws are:
1. LATE: Var(τ̂) ∝ 1/(Nπ²), so SD(τ̂) ∝ (Nπ²)^(-1/2)
2. PSA: Var(τ̂_PSA) ∝ 1/(Nπ), so SD(τ̂_PSA) ∝ (Nπ)^(-1/2)
3. Ghost Ads: Var(θ̂_GA) ∝ 1/(Nπ), so SD(θ̂_GA) ∝ (Nπ)^(-1/2)
4. PGA: Var(τ̂_PGA) ∝ 1/(Nπ) when p̂ ≈ 1, so SD(τ̂_PGA) ∝ (Nπ)^(-1/2)

Key predictions:
- LATE variance is inflated by factor 1/π compared to PSA/GA/PGA
- PSA, Ghost Ads, and PGA have identical variance scaling
- Doubling N should halve variance
- Halving π should double variance for PSA/GA/PGA, quadruple for LATE

Simulation Design:
------------------
- Vary sample size N ∈ {500, 1000, 2000, 5000}
- Vary compliance rate π ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- True treatment effect τ = 2.0
- Monte Carlo iterations: 10,000 per configuration
- Calculate empirical variance for each estimator
- Compare to theoretical scaling

Expected Results:
-----------------
1. log(SD) vs log(N): slope ≈ -0.5 for all estimators
2. log(SD) vs log(π): slope ≈ -0.5 for PSA/GA/PGA, slope ≈ -1.0 for LATE
3. Var(LATE) / Var(GA) ≈ 1/π
"""

import numpy as np
from tqdm import tqdm
import sys
from common_dgp import GhostAdsDGP

def run_variance_scaling_simulation():
    """Run simulation to verify variance scaling laws."""

    # Simulation parameters
    tau = 2.0
    sigma_0 = 2.0
    sigma_1 = 2.0
    p_predict = 0.97
    n_sims = 10000
    seed = 42

    # Vary N and π
    N_values = [500, 1000, 2000, 5000]
    pi_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("=" * 80)
    print("VARIANCE SCALING SIMULATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sample sizes N: {N_values}")
    print(f"  Compliance rates π: {pi_values}")
    print(f"  True treatment effect τ: {tau}")
    print(f"  Control outcome SD σ_0: {sigma_0}")
    print(f"  Treatment outcome SD σ_1: {sigma_1}")
    print(f"  PGA prediction accuracy: {p_predict}")
    print(f"  Monte Carlo iterations: {n_sims}")
    print(f"  Random seed: {seed}")
    print("\n" + "=" * 80)

    # Results storage
    results = []

    # Grid search over N and π
    total_configs = len(N_values) * len(pi_values)
    config_num = 0

    for N in N_values:
        for pi in pi_values:
            config_num += 1
            print(f"\n\nConfiguration {config_num}/{total_configs}: N={N}, π={pi}")

            # Storage for this configuration
            late_estimates = []
            psa_estimates = []
            ga_estimates = []
            pga_estimates = []

            # Monte Carlo loop
            for i in tqdm(range(n_sims), desc=f"N={N}, π={pi}", file=sys.stdout):
                dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0,
                                  sigma_1=None, p_predict=p_predict, seed=seed + i + config_num * 100000)

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

            # Calculate empirical variance and SD
            var_late = np.var(late_valid, ddof=1)
            var_psa = np.var(psa_valid, ddof=1)
            var_ga = np.var(ga_valid, ddof=1)
            var_pga = np.var(pga_valid, ddof=1)

            sd_late = np.sqrt(var_late)
            sd_psa = np.sqrt(var_psa)
            sd_ga = np.sqrt(var_ga)
            sd_pga = np.sqrt(var_pga)

            # Store results
            results.append({
                'N': N,
                'pi': pi,
                'var_late': var_late,
                'var_psa': var_psa,
                'var_ga': var_ga,
                'var_pga': var_pga,
                'sd_late': sd_late,
                'sd_psa': sd_psa,
                'sd_ga': sd_ga,
                'sd_pga': sd_pga,
                'n_valid_late': len(late_valid),
                'n_valid_psa': len(psa_valid),
                'n_valid_ga': len(ga_valid),
                'n_valid_pga': len(pga_valid)
            })

            print(f"\n  Empirical Standard Deviations:")
            print(f"    LATE: {sd_late:.6f}")
            print(f"    PSA: {sd_psa:.6f}")
            print(f"    Ghost Ads: {sd_ga:.6f}")
            print(f"    PGA: {sd_pga:.6f}")

    # Analysis: Variance scaling with N (for fixed π = 0.5)
    print("\n\n" + "=" * 80)
    print("ANALYSIS 1: VARIANCE SCALING WITH SAMPLE SIZE N (π = 0.5)")
    print("=" * 80)

    pi_fixed = 0.5
    results_fixed_pi = [r for r in results if r['pi'] == pi_fixed]

    print(f"\n{'N':<10} {'SD(LATE)':<15} {'SD(PSA)':<15} {'SD(GA)':<15} {'SD(PGA)':<15}")
    print("-" * 80)
    for r in results_fixed_pi:
        print(f"{r['N']:<10} {r['sd_late']:<15.6f} {r['sd_psa']:<15.6f} "
              f"{r['sd_ga']:<15.6f} {r['sd_pga']:<15.6f}")

    # Check log-log relationship: log(SD) = const - 0.5 * log(N)
    N_array = np.array([r['N'] for r in results_fixed_pi])
    log_N = np.log(N_array)

    for estimator in ['LATE', 'PSA', 'GA', 'PGA']:
        sd_key = f'sd_{estimator.lower().replace(" ", "_")}'
        sd_array = np.array([r[sd_key] for r in results_fixed_pi])
        log_sd = np.log(sd_array)

        # OLS regression
        slope = np.sum((log_N - np.mean(log_N)) * (log_sd - np.mean(log_sd))) / \
                np.sum((log_N - np.mean(log_N))**2)
        intercept = np.mean(log_sd) - slope * np.mean(log_N)
        r_squared = np.corrcoef(log_N, log_sd)[0, 1]**2

        print(f"\n{estimator}:")
        print(f"  log(SD) = {intercept:.4f} + {slope:.4f} * log(N)")
        print(f"  R² = {r_squared:.6f}")
        print(f"  Expected slope: -0.5")
        print(f"  Difference from theory: {abs(slope + 0.5):.4f}")

    # Analysis: Variance scaling with π (for fixed N = 2000)
    print("\n\n" + "=" * 80)
    print("ANALYSIS 2: VARIANCE SCALING WITH COMPLIANCE RATE π (N = 2000)")
    print("=" * 80)

    N_fixed = 2000
    results_fixed_N = [r for r in results if r['N'] == N_fixed]

    print(f"\n{'π':<10} {'SD(LATE)':<15} {'SD(PSA)':<15} {'SD(GA)':<15} {'SD(PGA)':<15}")
    print("-" * 80)
    for r in results_fixed_N:
        print(f"{r['pi']:<10.2f} {r['sd_late']:<15.6f} {r['sd_psa']:<15.6f} "
              f"{r['sd_ga']:<15.6f} {r['sd_pga']:<15.6f}")

    # Check log-log relationship
    pi_array = np.array([r['pi'] for r in results_fixed_N])
    log_pi = np.log(pi_array)

    print("\n\nLog-log regressions:")
    for estimator, expected_slope in [('LATE', -1.0), ('PSA', -0.5), ('GA', -0.5), ('PGA', -0.5)]:
        sd_key = f'sd_{estimator.lower().replace(" ", "_")}'
        sd_array = np.array([r[sd_key] for r in results_fixed_N])
        log_sd = np.log(sd_array)

        # OLS regression
        slope = np.sum((log_pi - np.mean(log_pi)) * (log_sd - np.mean(log_sd))) / \
                np.sum((log_pi - np.mean(log_pi))**2)
        intercept = np.mean(log_sd) - slope * np.mean(log_pi)
        r_squared = np.corrcoef(log_pi, log_sd)[0, 1]**2

        print(f"\n{estimator}:")
        print(f"  log(SD) = {intercept:.4f} + {slope:.4f} * log(π)")
        print(f"  R² = {r_squared:.6f}")
        print(f"  Expected slope: {expected_slope}")
        print(f"  Difference from theory: {abs(slope - expected_slope):.4f}")

    # Analysis: Variance ratio LATE vs Ghost Ads
    print("\n\n" + "=" * 80)
    print("ANALYSIS 3: VARIANCE RATIO Var(LATE) / Var(GA) ≈ 1/π")
    print("=" * 80)

    print(f"\n{'π':<10} {'Var(LATE)':<15} {'Var(GA)':<15} {'Ratio':<15} {'Theory (1/π)':<15} {'Match?':<10}")
    print("-" * 80)

    for r in results_fixed_N:
        ratio = r['var_late'] / r['var_ga']
        theory = 1 / r['pi']
        match = "✓" if abs(ratio - theory) / theory < 0.15 else "✗"  # 15% tolerance
        print(f"{r['pi']:<10.2f} {r['var_late']:<15.6f} {r['var_ga']:<15.6f} "
              f"{ratio:<15.6f} {theory:<15.6f} {match:<10}")

    # Analysis: PSA, GA, PGA variance consistency
    print("\n\n" + "=" * 80)
    print("ANALYSIS 4: PSA, GHOST ADS, AND PGA VARIANCE CONSISTENCY")
    print("=" * 80)

    print(f"\n{'N':<10} {'π':<10} {'SD(PSA)':<15} {'SD(GA)':<15} {'SD(PGA)':<15} {'Max Diff':<15}")
    print("-" * 80)

    for r in results:
        max_diff = max(abs(r['sd_psa'] - r['sd_ga']),
                       abs(r['sd_psa'] - r['sd_pga']),
                       abs(r['sd_ga'] - r['sd_pga']))
        print(f"{r['N']:<10} {r['pi']:<10.2f} {r['sd_psa']:<15.6f} {r['sd_ga']:<15.6f} "
              f"{r['sd_pga']:<15.6f} {max_diff:<15.6f}")

    # Verification
    print("\n\n" + "=" * 80)
    print("VERIFICATION: VARIANCE SCALING LAWS")
    print("=" * 80)

    # Check all slope conditions
    checks = []

    # 1. SD ∝ N^(-1/2) for all estimators at π = 0.5
    print("\n1. Checking SD ∝ N^(-1/2) for all estimators (π = 0.5):")
    for estimator, expected_slope in [('LATE', -0.5), ('PSA', -0.5), ('GA', -0.5), ('PGA', -0.5)]:
        sd_key = f'sd_{estimator.lower().replace(" ", "_")}'
        sd_array = np.array([r[sd_key] for r in results_fixed_pi])
        log_sd = np.log(sd_array)
        log_N_arr = np.log(np.array([r['N'] for r in results_fixed_pi]))

        slope = np.sum((log_N_arr - np.mean(log_N_arr)) * (log_sd - np.mean(log_sd))) / \
                np.sum((log_N_arr - np.mean(log_N_arr))**2)

        check = abs(slope - expected_slope) < 0.1
        checks.append(check)
        status = "✓" if check else "✗"
        print(f"  {status} {estimator}: slope = {slope:.4f} (expected {expected_slope})")

    # 2. SD ∝ π^(-1) for LATE, π^(-0.5) for PSA/GA/PGA at N = 2000
    print("\n2. Checking SD ∝ π^(expected) for all estimators (N = 2000):")
    for estimator, expected_slope in [('LATE', -1.0), ('PSA', -0.5), ('GA', -0.5), ('PGA', -0.5)]:
        sd_key = f'sd_{estimator.lower().replace(" ", "_")}'
        sd_array = np.array([r[sd_key] for r in results_fixed_N])
        log_sd = np.log(sd_array)
        log_pi_arr = np.log(np.array([r['pi'] for r in results_fixed_N]))

        slope = np.sum((log_pi_arr - np.mean(log_pi_arr)) * (log_sd - np.mean(log_sd))) / \
                np.sum((log_pi_arr - np.mean(log_pi_arr))**2)

        check = abs(slope - expected_slope) < 0.15
        checks.append(check)
        status = "✓" if check else "✗"
        print(f"  {status} {estimator}: slope = {slope:.4f} (expected {expected_slope})")

    # 3. Var(LATE) / Var(GA) ≈ 1/π
    print("\n3. Checking Var(LATE) / Var(GA) ≈ 1/π (N = 2000):")
    ratios_match = []
    for r in results_fixed_N:
        ratio = r['var_late'] / r['var_ga']
        theory = 1 / r['pi']
        match = abs(ratio - theory) / theory < 0.15
        ratios_match.append(match)
        status = "✓" if match else "✗"
        print(f"  {status} π = {r['pi']:.2f}: ratio = {ratio:.4f}, theory = {theory:.4f}")

    checks.extend(ratios_match)

    # Final verdict
    print("\n" + "=" * 80)
    if all(checks):
        print("✓ SUCCESS: All variance scaling laws confirmed")
        print("✓ LATE variance scales as 1/(Nπ²)")
        print("✓ PSA/GA/PGA variance scales as 1/(Nπ)")
        print("✓ Var(LATE) / Var(GA) ≈ 1/π")
    else:
        print("✗ PARTIAL SUCCESS: Some variance scaling laws not perfectly confirmed")
        print(f"  {sum(checks)}/{len(checks)} checks passed")

    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_variance_scaling_simulation()
