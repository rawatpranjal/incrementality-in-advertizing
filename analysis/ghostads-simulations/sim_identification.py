"""
Simulation: Unbiased Identification

Verifies Appendices A.2-A.9: All estimators are unbiased under their assumptions

Theory:
-------
1. ITT (A.2): E[τ̂_ITT] = τ_ITT = π · τ_ATT
2. LATE (A.4): E[τ̂_LATE] = τ_ATT (under one-sided noncompliance)
3. PSA (A.5): E[τ̂_PSA] = τ_ATT (under perfect blind and no placebo effect)
4. Ghost Ads (A.7): E[θ̂_GA] = τ_ATT (under counterfactual tagging)
5. PGA (A.9): E[τ̂_PGA] = LATE_PGA ≈ τ_ATT (under predetermined prediction)

Simulation Design:
------------------
- Sample size N = 2000
- Compliance rate π = 0.5
- True treatment effect τ = 2.0 (constant across all units, so ATT = τ)
- Monte Carlo iterations: 10,000
- Calculate empirical bias: E[τ̂] - τ_true

Expected Results:
-----------------
1. All estimators should have empirical bias ≈ 0
2. ITT should be unbiased for π · τ (not τ)
3. LATE, PSA, GA, PGA should all be unbiased for τ_ATT = τ
"""

import numpy as np
from tqdm import tqdm
import sys
from common_dgp import GhostAdsDGP

def run_identification_simulation():
    """Run simulation to verify unbiased identification."""

    # Simulation parameters
    N = 2000
    pi = 0.5
    tau = 2.0
    sigma_0 = 2.0
    sigma_1 = 2.0
    p_predict = 0.97
    n_sims = 10000
    seed = 42

    print("=" * 80)
    print("IDENTIFICATION SIMULATION: UNBIASED ESTIMATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sample size N: {N}")
    print(f"  Compliance rate π: {pi}")
    print(f"  True treatment effect τ: {tau}")
    print(f"  Control outcome SD σ_0: {sigma_0}")
    print(f"  Treatment outcome SD σ_1: {sigma_1}")
    print(f"  PGA prediction accuracy: {p_predict}")
    print(f"  Monte Carlo iterations: {n_sims}")
    print(f"  Random seed: {seed}")
    print("\n" + "=" * 80)

    # VERIFY DGP MOMENTS FIRST
    print("\nVERIFYING DGP MOMENTS (N=100,000)...")
    print("-" * 80)

    # Generate large sample to check DGP produces correct moments
    dgp_test = GhostAdsDGP(N=100000, pi=pi, tau=tau, sigma_0=sigma_0,
                           sigma_1=None, p_predict=p_predict, seed=999)

    print(f"E[Y₀] = {np.mean(dgp_test.Y0):.6f} (expected: 0.000000)")
    print(f"SD[Y₀] = {np.std(dgp_test.Y0, ddof=1):.6f} (expected: {sigma_0:.6f})")
    print(f"E[Y₁] = {np.mean(dgp_test.Y1):.6f} (expected: {tau:.6f})")
    print(f"SD[Y₁] = {np.std(dgp_test.Y1, ddof=1):.6f} (expected: {sigma_0:.6f})")
    print(f"P(D=1|Z=1) = {np.mean(dgp_test.D_potential_1):.6f} (expected: {pi:.6f})")
    print(f"ATT = {np.mean(dgp_test.tau_i[dgp_test.D_potential_1 == 1]):.6f} (expected: {tau:.6f})")
    print(f"ITT = {pi * tau:.6f} (expected: {pi * tau:.6f})")

    # Check if moments are within acceptable range
    moment_checks = [
        abs(np.mean(dgp_test.Y0)) < 0.01,  # E[Y0] ≈ 0
        abs(np.std(dgp_test.Y0, ddof=1) - sigma_0) < 0.01,  # SD[Y0] ≈ sigma_0
        abs(np.mean(dgp_test.Y1) - tau) < 0.01,  # E[Y1] ≈ tau
        abs(np.mean(dgp_test.D_potential_1) - pi) < 0.01,  # P(D=1) ≈ pi
    ]

    if all(moment_checks):
        print("\n✓ All DGP moments verified!")
    else:
        print("\n✗ WARNING: Some DGP moments deviate from expected values")

    print("=" * 80)

    # Storage
    itt_estimates = []
    late_estimates = []
    psa_estimates = []
    ga_estimates = []
    pga_estimates = []

    print("\nRunning Monte Carlo simulation...")

    # Monte Carlo loop
    for i in tqdm(range(n_sims), desc="Simulations", file=sys.stdout):
        dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0,
                          sigma_1=None, p_predict=p_predict, seed=seed + i)

        # Estimate all estimators
        itt_res = dgp.estimate_itt()
        itt_estimates.append(itt_res['estimate'])

        late_res = dgp.estimate_late()
        late_estimates.append(late_res['estimate'])

        psa_res = dgp.estimate_psa()
        psa_estimates.append(psa_res['estimate'])

        ga_res = dgp.estimate_ghost_ads()
        ga_estimates.append(ga_res['estimate'])

        pga_res = dgp.estimate_pga()
        pga_estimates.append(pga_res['estimate'])

    # Convert to arrays
    itt_estimates = np.array(itt_estimates)
    late_estimates = np.array(late_estimates)
    psa_estimates = np.array(psa_estimates)
    ga_estimates = np.array(ga_estimates)
    pga_estimates = np.array(pga_estimates)

    # Remove NaN values
    late_estimates_valid = late_estimates[~np.isnan(late_estimates)]
    psa_estimates_valid = psa_estimates[~np.isnan(psa_estimates)]
    ga_estimates_valid = ga_estimates[~np.isnan(ga_estimates)]
    pga_estimates_valid = pga_estimates[~np.isnan(pga_estimates)]

    # True effects
    tau_itt_true = pi * tau
    tau_att_true = tau  # ATT = τ since effect is constant

    # Calculate statistics for each estimator
    results = {
        'ITT': {
            'mean': np.mean(itt_estimates),
            'sd': np.std(itt_estimates, ddof=1),
            'true': tau_itt_true,
            'bias': np.mean(itt_estimates) - tau_itt_true,
            'rmse': np.sqrt(np.mean((itt_estimates - tau_itt_true)**2)),
            'n_valid': len(itt_estimates)
        },
        'LATE': {
            'mean': np.mean(late_estimates_valid),
            'sd': np.std(late_estimates_valid, ddof=1),
            'true': tau_att_true,
            'bias': np.mean(late_estimates_valid) - tau_att_true,
            'rmse': np.sqrt(np.mean((late_estimates_valid - tau_att_true)**2)),
            'n_valid': len(late_estimates_valid)
        },
        'PSA': {
            'mean': np.mean(psa_estimates_valid),
            'sd': np.std(psa_estimates_valid, ddof=1),
            'true': tau_att_true,
            'bias': np.mean(psa_estimates_valid) - tau_att_true,
            'rmse': np.sqrt(np.mean((psa_estimates_valid - tau_att_true)**2)),
            'n_valid': len(psa_estimates_valid)
        },
        'Ghost Ads': {
            'mean': np.mean(ga_estimates_valid),
            'sd': np.std(ga_estimates_valid, ddof=1),
            'true': tau_att_true,
            'bias': np.mean(ga_estimates_valid) - tau_att_true,
            'rmse': np.sqrt(np.mean((ga_estimates_valid - tau_att_true)**2)),
            'n_valid': len(ga_estimates_valid)
        },
        'PGA': {
            'mean': np.mean(pga_estimates_valid),
            'sd': np.std(pga_estimates_valid, ddof=1),
            'true': tau_att_true,
            'bias': np.mean(pga_estimates_valid) - tau_att_true,
            'rmse': np.sqrt(np.mean((pga_estimates_valid - tau_att_true)**2)),
            'n_valid': len(pga_estimates_valid)
        }
    }

    # Print results
    print("\n\n" + "=" * 80)
    print("RESULTS: UNBIASED IDENTIFICATION")
    print("=" * 80)

    for estimator, stats in results.items():
        print(f"\n{estimator}:")
        print(f"  Empirical mean: {stats['mean']:.6f}")
        print(f"  Standard deviation: {stats['sd']:.6f}")
        print(f"  True effect: {stats['true']:.6f}")
        print(f"  Bias: {stats['bias']:.6f}")
        print(f"  RMSE: {stats['rmse']:.6f}")
        print(f"  Valid samples: {stats['n_valid']} / {n_sims}")

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE: BIAS COMPARISON")
    print("=" * 80)
    print(f"\n{'Estimator':<15} {'Mean':<12} {'True':<12} {'Bias':<12} {'Bias %':<12} {'Unbiased?':<12}")
    print("-" * 80)

    for estimator, stats in results.items():
        bias_pct = 100 * stats['bias'] / stats['true'] if stats['true'] != 0 else 0
        unbiased = "✓" if abs(stats['bias']) < 0.05 else "✗"
        print(f"{estimator:<15} {stats['mean']:<12.6f} {stats['true']:<12.6f} "
              f"{stats['bias']:<12.6f} {bias_pct:<12.3f} {unbiased:<12}")

    # Verification
    print("\n\n" + "=" * 80)
    print("VERIFICATION: UNBIASED IDENTIFICATION")
    print("=" * 80)

    # Check if all estimators are unbiased (bias < 0.05)
    all_unbiased = all(abs(stats['bias']) < 0.05 for stats in results.values())

    if all_unbiased:
        print("\n✓ SUCCESS: All estimators are unbiased (|bias| < 0.05)")
        print("\nDetailed verification:")
        print(f"  ✓ ITT is unbiased for τ_ITT = π · τ = {tau_itt_true:.2f}")
        print(f"  ✓ LATE is unbiased for τ_ATT = {tau_att_true:.2f}")
        print(f"  ✓ PSA is unbiased for τ_ATT = {tau_att_true:.2f}")
        print(f"  ✓ Ghost Ads is unbiased for τ_ATT = {tau_att_true:.2f}")
        print(f"  ✓ PGA is unbiased for τ_ATT = {tau_att_true:.2f}")
    else:
        print("\n✗ FAILURE: Some estimators show significant bias")
        for estimator, stats in results.items():
            if abs(stats['bias']) >= 0.05:
                print(f"  ✗ {estimator}: bias = {stats['bias']:.6f}")

    # Check that ITT identifies the diluted effect
    print("\n\nSpecial check: ITT dilution")
    itt_targets_att = abs(results['ITT']['mean'] - tau_att_true) < 0.05
    itt_targets_itt = abs(results['ITT']['mean'] - tau_itt_true) < 0.05

    if itt_targets_itt and not itt_targets_att:
        print(f"  ✓ ITT correctly identifies τ_ITT = π · τ = {tau_itt_true:.2f}")
        print(f"  ✓ ITT does NOT identify τ_ATT = {tau_att_true:.2f} (as expected)")
    else:
        print(f"  ✗ ITT identification issue")

    # Check that LATE recovers ATT from diluted ITT
    print("\n\nCheck: LATE recovery of ATT from diluted ITT")
    late_recovers = abs(results['LATE']['mean'] - tau_att_true) < 0.05
    if late_recovers:
        print(f"  ✓ LATE successfully recovers τ_ATT = {tau_att_true:.2f} from diluted ITT")
        print(f"  ✓ This confirms the Wald estimator corrects for dilution")
    else:
        print(f"  ✗ LATE does not recover ATT correctly")

    # Check that PSA, GA, PGA all identify the same effect
    print("\n\nCheck: PSA, Ghost Ads, and PGA consistency")
    means = [results['PSA']['mean'], results['Ghost Ads']['mean'], results['PGA']['mean']]
    mean_diff = np.max(means) - np.min(means)
    if mean_diff < 0.05:
        print(f"  ✓ PSA, Ghost Ads, and PGA all identify the same effect")
        print(f"    PSA: {results['PSA']['mean']:.4f}")
        print(f"    Ghost Ads: {results['Ghost Ads']['mean']:.4f}")
        print(f"    PGA: {results['PGA']['mean']:.4f}")
        print(f"    Max difference: {mean_diff:.6f}")
    else:
        print(f"  ✗ PSA, Ghost Ads, and PGA show inconsistent estimates")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_identification_simulation()
