"""
Section 4: Multi-Period Budget Pacing
Compares No Pacing, Uniform, Adaptive, and Lift-Aware Pacing
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

VALUE_PER_CONVERSION = 100.0


def solve_allocation(causal_score, c, budget_remaining, integrality=True):
    """
    Solve MILP/LP for user allocation given budget
    Returns: selected users (binary or fractional)
    """
    M = len(causal_score)
    c_obj = -causal_score
    A = c.reshape(1, -1)
    b_ub = np.array([budget_remaining])
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))

    if integrality:
        result = milp(
            c=c_obj,
            constraints=LinearConstraint(A, -np.inf, b_ub),
            bounds=bounds,
            integrality=np.ones(M),
            options={'disp': False}
        )
    else:
        from scipy.optimize import linprog
        result = linprog(
            c=c_obj,
            A_ub=A,
            b_ub=b_ub,
            bounds=(0, 1),
            method='highs',
            options={'disp': False}
        )

    return result.x if result.success else np.zeros(M)


def run_simulation(seed=42):
    """
    Simulate multi-period budget pacing strategies
    Returns: dict with results for each pacing method
    """
    np.random.seed(seed)

    # Data generation - users arrive over time with varying lift
    M = 1000
    B = 20.0
    T = 24  # 24 hours

    # Generate hourly user pools with time-varying lift
    # Peak hours (8am, 12pm, 6pm) have higher lift users
    peak_hours = [8, 12, 18]
    users_per_hour = []

    for t in range(T):
        n_users = int(M / T * (1.5 if t in peak_hours else 1.0))

        p_0 = np.random.beta(2, 8, n_users)
        epsilon = np.random.normal(0, 0.005, n_users)

        # Peak hours have 30% higher lift
        lift_multiplier = 1.3 if t in peak_hours else 1.0
        tau = (0.08 / (1 + 10 * p_0) + epsilon) * lift_multiplier
        tau = np.clip(tau, 0.001, 0.15)

        ctr = np.random.beta(3, 20, n_users)
        cpc = np.random.exponential(0.5, n_users) + 0.1
        c = ctr * cpc

        causal_score = ctr * tau * VALUE_PER_CONVERSION

        users_per_hour.append({
            'causal_score': causal_score,
            'cost': c,
            'tau': tau,
            'hour': t
        })

    results = {}

    # 1. NO PACING (Greedy - spend all in first hour)
    start = time.time()
    total_obj = 0
    total_budget_used = 0
    hours_depleted = 0

    budget_remaining = B
    for t in range(T):
        if budget_remaining <= 0.01:
            hours_depleted = t
            break

        users = users_per_hour[t]
        x_selected = solve_allocation(
            users['causal_score'],
            users['cost'],
            budget_remaining,
            integrality=True
        )

        spent = (x_selected * users['cost']).sum()
        total_obj += (x_selected * users['causal_score']).sum()
        total_budget_used += spent
        budget_remaining -= spent

        if budget_remaining <= 0.01:
            hours_depleted = t + 1
            break

    iroas_greedy = total_obj / total_budget_used if total_budget_used > 0 else 0

    results['No Pacing'] = {
        'utilization': total_budget_used / B * 100,
        'hours_depleted': hours_depleted,
        'iroas': iroas_greedy,
        'objective': total_obj,
        'runtime': time.time() - start
    }

    # 2. UNIFORM PACING (equal budget per hour)
    start = time.time()
    budget_per_hour = B / T
    total_obj = 0
    total_budget_used = 0

    for t in range(T):
        users = users_per_hour[t]
        x_selected = solve_allocation(
            users['causal_score'],
            users['cost'],
            budget_per_hour,
            integrality=True
        )

        spent = (x_selected * users['cost']).sum()
        total_obj += (x_selected * users['causal_score']).sum()
        total_budget_used += spent

    iroas_uniform = total_obj / total_budget_used if total_budget_used > 0 else 0

    results['Uniform Pacing'] = {
        'utilization': total_budget_used / B * 100,
        'hours_depleted': T,
        'iroas': iroas_uniform,
        'objective': total_obj,
        'runtime': time.time() - start
    }

    # 3. ADAPTIVE PACING (allocate more to peak hours)
    start = time.time()
    total_obj = 0
    total_budget_used = 0

    # Allocate 50% more budget to peak hours
    budget_per_hour_adaptive = np.ones(T) * (B / T)
    for t in peak_hours:
        budget_per_hour_adaptive[t] *= 1.5
    # Normalize to sum to B
    budget_per_hour_adaptive = budget_per_hour_adaptive / budget_per_hour_adaptive.sum() * B

    for t in range(T):
        users = users_per_hour[t]
        x_selected = solve_allocation(
            users['causal_score'],
            users['cost'],
            budget_per_hour_adaptive[t],
            integrality=True
        )

        spent = (x_selected * users['cost']).sum()
        total_obj += (x_selected * users['causal_score']).sum()
        total_budget_used += spent

    iroas_adaptive = total_obj / total_budget_used if total_budget_used > 0 else 0

    results['Adaptive Pacing'] = {
        'utilization': total_budget_used / B * 100,
        'hours_depleted': T,
        'iroas': iroas_adaptive,
        'objective': total_obj,
        'runtime': time.time() - start
    }

    # 4. LIFT-AWARE PACING (allocate proportional to expected lift)
    start = time.time()

    # Estimate expected lift per hour from historical data
    expected_lift_per_hour = np.array([users_per_hour[t]['tau'].mean() for t in range(T)])
    budget_per_hour_liftaware = (expected_lift_per_hour / expected_lift_per_hour.sum()) * B

    total_obj = 0
    total_budget_used = 0

    for t in range(T):
        users = users_per_hour[t]
        x_selected = solve_allocation(
            users['causal_score'],
            users['cost'],
            budget_per_hour_liftaware[t],
            integrality=True
        )

        spent = (x_selected * users['cost']).sum()
        total_obj += (x_selected * users['causal_score']).sum()
        total_budget_used += spent

    iroas_liftaware = total_obj / total_budget_used if total_budget_used > 0 else 0

    results['Lift-Aware Pacing'] = {
        'utilization': total_budget_used / B * 100,
        'hours_depleted': T,
        'iroas': iroas_liftaware,
        'objective': total_obj,
        'runtime': time.time() - start
    }

    return results


if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("SECTION 4: MULTI-PERIOD BUDGET PACING")
    print("="*80)

    # Run multiple times
    n_runs = 10
    all_results = []

    for seed in range(42, 42 + n_runs):
        all_results.append(run_simulation(seed))

    # Aggregate results
    methods = ['No Pacing', 'Uniform Pacing', 'Adaptive Pacing', 'Lift-Aware Pacing']

    summary = []
    for method in methods:
        utilizations = [r[method]['utilization'] for r in all_results]
        hours = [r[method]['hours_depleted'] for r in all_results]
        iroas_vals = [r[method]['iroas'] for r in all_results]
        objectives = [r[method]['objective'] for r in all_results]

        summary.append({
            'Pacing Strategy': method,
            'Budget Utilization': f"{np.mean(utilizations):.1f}%",
            'Hours Active': f"{np.mean(hours):.1f}",
            'iROAS': f"{np.mean(iroas_vals):.2f}×",
            'Objective Value': f"${np.mean(objectives):.2f}"
        })

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    # Calculate improvement
    baseline_iroas = np.mean([r['No Pacing']['iroas'] for r in all_results])
    liftaware_iroas = np.mean([r['Lift-Aware Pacing']['iroas'] for r in all_results])
    improvement = (liftaware_iroas - baseline_iroas) / baseline_iroas * 100

    print(f"\nImprovement (Lift-Aware vs No Pacing): {improvement:.1f}%")
    print(f"Average runtime per strategy: {np.mean([np.mean([r[m]['runtime'] for m in methods]) for r in all_results]):.3f}s")
