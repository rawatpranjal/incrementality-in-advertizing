"""
Section 3: Fractional Relaxation
Compares MILP, LP, and Greedy Rounding
"""

import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
import time

VALUE_PER_CONVERSION = 100.0


def run_simulation(seed=42):
    """
    Simulate fractional relaxation vs binary MILP
    Returns: dict with results for MILP, LP, Greedy
    """
    np.random.seed(seed)

    # Same data generation as Section 2
    M = 1000
    B = 20.0

    p_0 = np.random.beta(2, 8, M)
    epsilon = np.random.normal(0, 0.005, M)
    tau = 0.08 / (1 + 10 * p_0) + epsilon
    tau = np.clip(tau, 0.001, 0.15)
    ctr = np.random.beta(3, 20, M)
    cpc = np.random.exponential(0.5, M) + 0.1
    c = ctr * cpc
    v = VALUE_PER_CONVERSION

    # Causal scores
    causal_score = ctr * tau * v

    results = {}

    # 1. BINARY MILP (exact solution)
    start = time.time()
    c_obj = -causal_score
    A = c.reshape(1, -1)
    b_ub = np.array([B])
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))

    result_milp = milp(
        c=c_obj,
        constraints=LinearConstraint(A, -np.inf, b_ub),
        bounds=bounds,
        integrality=np.ones(M),
        options={'disp': False}
    )
    milp_time = time.time() - start

    x_milp = result_milp.x
    obj_milp = -result_milp.fun

    results['Binary MILP'] = {
        'objective': obj_milp,
        'fractional_users': 0,
        'runtime': milp_time,
        'gap': 0.0
    }

    # 2. FRACTIONAL LP (relaxed solution)
    start = time.time()
    result_lp = linprog(
        c=c_obj,
        A_ub=A,
        b_ub=b_ub,
        bounds=(0, 1),
        method='highs',
        options={'disp': False}
    )
    lp_time = time.time() - start

    x_lp = result_lp.x
    obj_lp = -result_lp.fun
    n_fractional = np.sum((x_lp > 0.001) & (x_lp < 0.999))

    results['Fractional LP'] = {
        'objective': obj_lp,
        'fractional_users': n_fractional,
        'runtime': lp_time,
        'gap': (obj_lp - obj_milp) / obj_milp * 100
    }

    # 3. GREEDY ROUNDING (practical approximation)
    start = time.time()
    # Sort by bang-for-buck ratio
    ratios = causal_score / c
    sorted_idx = np.argsort(-ratios)

    x_greedy = np.zeros(M)
    budget_used = 0

    for idx in sorted_idx:
        if budget_used + c[idx] <= B:
            x_greedy[idx] = 1
            budget_used += c[idx]

    greedy_time = time.time() - start
    obj_greedy = (x_greedy * causal_score).sum()

    results['Greedy Rounding'] = {
        'objective': obj_greedy,
        'fractional_users': 0,
        'runtime': greedy_time,
        'gap': (obj_greedy - obj_milp) / obj_milp * 100
    }

    return results


if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("SECTION 3: FRACTIONAL RELAXATION")
    print("="*80)

    # Run multiple times
    n_runs = 10
    all_results = []

    for seed in range(42, 42 + n_runs):
        all_results.append(run_simulation(seed))

    # Aggregate results
    methods = ['Binary MILP', 'Fractional LP', 'Greedy Rounding']

    summary = []
    for method in methods:
        objectives = [r[method]['objective'] for r in all_results]
        frac_users = [r[method]['fractional_users'] for r in all_results]
        runtimes = [r[method]['runtime'] for r in all_results]
        gaps = [r[method]['gap'] for r in all_results]

        summary.append({
            'Solution Method': method,
            'Objective Value': f"${np.mean(objectives):.2f}",
            'Fractional Users': f"{int(np.mean(frac_users))}",
            'Runtime': f"{np.mean(runtimes):.2f}s",
            'Gap from Binary': f"{np.mean(gaps):+.2f}%"
        })

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    print(f"\nIntegrality gap: {abs(np.mean([r['Fractional LP']['gap'] for r in all_results])):.2f}%")
    print(f"Greedy speedup: {np.mean([r['Binary MILP']['runtime'] / r['Greedy Rounding']['runtime'] for r in all_results]):.0f}×")
