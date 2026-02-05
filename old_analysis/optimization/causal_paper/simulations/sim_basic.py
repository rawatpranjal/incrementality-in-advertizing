"""
Section 2: Standard vs Lift-Based Optimization
Matches paper Table 1 exactly
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

VALUE_PER_CONVERSION = 100.0  # Paper specification


def run_simulation(seed=42):
    """
    Simulate standard vs lift-based optimization
    Returns: dict with results for all methods
    """
    np.random.seed(seed)

    # Parameters from paper
    M = 1000  # users
    B = 20.0  # budget

    # Data generation as specified in paper
    p_0 = np.random.beta(2, 8, M)  # Baseline conversion probability
    epsilon = np.random.normal(0, 0.005, M)
    tau = 0.08 / (1 + 10 * p_0) + epsilon  # Treatment effect
    tau = np.clip(tau, 0.001, 0.15)
    ctr = np.random.beta(3, 20, M)  # Click-through rate
    cpc = np.random.exponential(0.5, M) + 0.1  # Cost per click
    c = ctr * cpc  # Cost per impression
    v = VALUE_PER_CONVERSION

    # Verify negative correlation
    correlation = np.corrcoef(p_0, tau)[0, 1]

    results = {'correlation': correlation}

    # 1. RANDOM SELECTION
    random_idx = np.random.choice(M, size=200, replace=False)
    results['Random Selection'] = {
        'users': 200,
        'spend': c[random_idx].sum(),
        'inc_value': (ctr[random_idx] * tau[random_idx] * v).sum(),
    }
    results['Random Selection']['iroas'] = (
        results['Random Selection']['inc_value'] / results['Random Selection']['spend']
    )

    # 2. STANDARD OPTIMIZATION (correlation-based)
    correlation_score = ctr * (p_0 + tau) * v
    c_obj = -correlation_score
    A = c.reshape(1, -1)
    b_ub = np.array([B])
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))
    integrality = np.ones(M)

    result = milp(
        c=c_obj,
        constraints=LinearConstraint(A, -np.inf, b_ub),
        bounds=bounds,
        integrality=integrality,
        options={'disp': False}
    )

    x_corr = result.x.astype(int)
    corr_idx = np.where(x_corr == 1)[0]
    results['Standard Optimization'] = {
        'users': len(corr_idx),
        'spend': c[corr_idx].sum(),
        'inc_value': (ctr[corr_idx] * tau[corr_idx] * v).sum(),
    }
    results['Standard Optimization']['iroas'] = (
        results['Standard Optimization']['inc_value'] / results['Standard Optimization']['spend']
    )

    # 3. LIFT-BASED OPTIMIZATION (causal)
    # Add realistic estimation noise
    noise_std = 0.3 * tau.std()
    tau_est = tau + np.random.normal(0, noise_std, M)
    tau_est = np.clip(tau_est, 0.001, 0.2)

    causal_score = ctr * tau_est * v
    c_obj = -causal_score

    result = milp(
        c=c_obj,
        constraints=LinearConstraint(A, -np.inf, b_ub),
        bounds=bounds,
        integrality=integrality,
        options={'disp': False}
    )

    x_causal = result.x.astype(int)
    causal_idx = np.where(x_causal == 1)[0]
    results['Lift-Based Optimization'] = {
        'users': len(causal_idx),
        'spend': c[causal_idx].sum(),
        'inc_value': (ctr[causal_idx] * tau[causal_idx] * v).sum(),
    }
    results['Lift-Based Optimization']['iroas'] = (
        results['Lift-Based Optimization']['inc_value'] / results['Lift-Based Optimization']['spend']
    )

    # 4. ORACLE (perfect information)
    oracle_score = ctr * tau * v
    c_obj = -oracle_score

    result = milp(
        c=c_obj,
        constraints=LinearConstraint(A, -np.inf, b_ub),
        bounds=bounds,
        integrality=integrality,
        options={'disp': False}
    )

    x_oracle = result.x.astype(int)
    oracle_idx = np.where(x_oracle == 1)[0]
    results['Perfect Information (Oracle)'] = {
        'users': len(oracle_idx),
        'spend': c[oracle_idx].sum(),
        'inc_value': (ctr[oracle_idx] * tau[oracle_idx] * v).sum(),
    }
    results['Perfect Information (Oracle)']['iroas'] = (
        results['Perfect Information (Oracle)']['inc_value'] / results['Perfect Information (Oracle)']['spend']
    )

    return results


if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("SECTION 2: STANDARD VS LIFT-BASED OPTIMIZATION")
    print("="*80)

    # Run multiple times for statistical validity
    n_runs = 10
    all_results = []

    for seed in range(42, 42 + n_runs):
        all_results.append(run_simulation(seed))

    # Aggregate results
    methods = ['Random Selection', 'Standard Optimization', 'Lift-Based Optimization', 'Perfect Information (Oracle)']

    summary = []
    for method in methods:
        users = [r[method]['users'] for r in all_results]
        spend = [r[method]['spend'] for r in all_results]
        inc_value = [r[method]['inc_value'] for r in all_results]
        iroas = [r[method]['iroas'] for r in all_results]

        summary.append({
            'Method': method,
            'Users Selected': f"{np.mean(users):.0f}",
            'Incremental Value': f"${np.mean(inc_value):.2f}",
            'Spend': f"${np.mean(spend):.2f}",
            'iROAS': f"{np.mean(iroas):.2f}×"
        })

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    # Calculate improvement
    standard_iroas = [r['Standard Optimization']['iroas'] for r in all_results]
    lift_iroas = [r['Lift-Based Optimization']['iroas'] for r in all_results]
    improvement = np.mean([(l/s - 1) * 100 for l, s in zip(lift_iroas, standard_iroas)])

    print(f"\nLift-based optimization achieves {improvement:.0f}% higher iROAS than standard methods")

    # Oracle performance
    oracle_iroas = [r['Perfect Information (Oracle)']['iroas'] for r in all_results]
    oracle_capture = np.mean([l/o * 100 for l, o in zip(lift_iroas, oracle_iroas)])
    print(f"Captures {oracle_capture:.1f}% of oracle performance")

    # Correlation
    correlations = [r['correlation'] for r in all_results]
    print(f"\nCorrelation(baseline, lift): {np.mean(correlations):.2f} ± {np.std(correlations):.2f}")
