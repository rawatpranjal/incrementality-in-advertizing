"""
Section 6: Cross-Platform Budget Allocation
Allocates budget optimally across multiple advertising platforms
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

VALUE_PER_CONVERSION = 100.0


def solve_platform_allocation(causal_scores_platform, costs_platform, budget_platform):
    """
    Solve MILP for single platform
    Returns: selected users, objective value
    """
    M = len(causal_scores_platform)
    c_obj = -causal_scores_platform
    A = costs_platform.reshape(1, -1)
    b_ub = np.array([budget_platform])
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))

    result = milp(
        c=c_obj,
        constraints=LinearConstraint(A, -np.inf, b_ub),
        bounds=bounds,
        integrality=np.ones(M),
        options={'disp': False}
    )

    if result.success:
        return result.x, -result.fun, (result.x * costs_platform).sum()
    else:
        return np.zeros(M), 0.0, 0.0


def run_simulation(seed=42):
    """
    Simulate cross-platform allocation strategies
    Platforms: Search, Display, Video
    Returns: dict with results for each allocation strategy
    """
    np.random.seed(seed)

    # Data generation
    M = 300  # users per platform (reduced for speed)
    B = 20.0  # total budget
    n_platforms = 3
    platform_names = ['Search', 'Display', 'Video']

    # Generate data for each platform with different characteristics
    platforms = {}

    # SEARCH: High intent, high cost, high conversion
    p_0_search = np.random.beta(3, 7, M)  # Higher baseline
    tau_search = 0.06 / (1 + 8 * p_0_search) + np.random.normal(0, 0.005, M)
    tau_search = np.clip(tau_search, 0.001, 0.12)
    ctr_search = np.random.beta(4, 16, M)  # Higher CTR
    cpc_search = np.random.exponential(0.7, M) + 0.3  # Higher cost
    c_search = ctr_search * cpc_search
    causal_score_search = ctr_search * tau_search * VALUE_PER_CONVERSION

    platforms['Search'] = {
        'causal_score': causal_score_search,
        'cost': c_search,
        'baseline': p_0_search,
        'lift': tau_search
    }

    # DISPLAY: Lower intent, low cost, moderate conversion
    p_0_display = np.random.beta(2, 10, M)  # Lower baseline
    tau_display = 0.10 / (1 + 12 * p_0_display) + np.random.normal(0, 0.005, M)
    tau_display = np.clip(tau_display, 0.001, 0.15)
    ctr_display = np.random.beta(2, 25, M)  # Lower CTR
    cpc_display = np.random.exponential(0.3, M) + 0.05  # Lower cost
    c_display = ctr_display * cpc_display
    causal_score_display = ctr_display * tau_display * VALUE_PER_CONVERSION

    platforms['Display'] = {
        'causal_score': causal_score_display,
        'cost': c_display,
        'baseline': p_0_display,
        'lift': tau_display
    }

    # VIDEO: Branding, medium cost, moderate conversion with higher variance
    p_0_video = np.random.beta(2.5, 8, M)
    tau_video = 0.08 / (1 + 10 * p_0_video) + np.random.normal(0, 0.007, M)
    tau_video = np.clip(tau_video, 0.001, 0.14)
    ctr_video = np.random.beta(3, 22, M)
    cpc_video = np.random.exponential(0.5, M) + 0.2
    c_video = ctr_video * cpc_video
    causal_score_video = ctr_video * tau_video * VALUE_PER_CONVERSION

    platforms['Video'] = {
        'causal_score': causal_score_video,
        'cost': c_video,
        'baseline': p_0_video,
        'lift': tau_video
    }

    results = {}

    # 1. SINGLE PLATFORM (allocate all to best platform)
    start = time.time()
    best_platform = None
    best_obj = 0

    for pname in platform_names:
        x, obj, spent = solve_platform_allocation(
            platforms[pname]['causal_score'],
            platforms[pname]['cost'],
            B
        )
        if obj > best_obj:
            best_obj = obj
            best_platform = pname

    iroas_single = best_obj / B if B > 0 else 0

    results['Single Platform'] = {
        'objective': best_obj,
        'iroas': iroas_single,
        'best_platform': best_platform,
        'budget_split': {pname: B if pname == best_platform else 0 for pname in platform_names},
        'runtime': time.time() - start
    }

    # 2. EQUAL SPLIT (divide budget equally)
    start = time.time()
    budget_per_platform = B / n_platforms
    total_obj = 0
    budget_splits = {}

    for pname in platform_names:
        x, obj, spent = solve_platform_allocation(
            platforms[pname]['causal_score'],
            platforms[pname]['cost'],
            budget_per_platform
        )
        total_obj += obj
        budget_splits[pname] = spent

    iroas_equal = total_obj / sum(budget_splits.values()) if sum(budget_splits.values()) > 0 else 0

    results['Equal Split'] = {
        'objective': total_obj,
        'iroas': iroas_equal,
        'budget_split': budget_splits,
        'runtime': time.time() - start
    }

    # 3. PROPORTIONAL ALLOCATION (weighted by platform efficiency)
    start = time.time()

    # Estimate efficiency per platform (iROAS per unit budget)
    efficiencies = {}
    for pname in platform_names:
        # Test with small budget
        test_budget = 0.1
        x, obj, spent = solve_platform_allocation(
            platforms[pname]['causal_score'],
            platforms[pname]['cost'],
            test_budget
        )
        efficiencies[pname] = obj / spent if spent > 0 else 0

    # Allocate proportional to efficiency
    total_efficiency = sum(efficiencies.values())
    budget_splits = {pname: B * (efficiencies[pname] / total_efficiency) for pname in platform_names}

    total_obj = 0
    actual_spent = {}
    for pname in platform_names:
        x, obj, spent = solve_platform_allocation(
            platforms[pname]['causal_score'],
            platforms[pname]['cost'],
            budget_splits[pname]
        )
        total_obj += obj
        actual_spent[pname] = spent

    iroas_proportional = total_obj / sum(actual_spent.values()) if sum(actual_spent.values()) > 0 else 0

    results['Proportional'] = {
        'objective': total_obj,
        'iroas': iroas_proportional,
        'budget_split': actual_spent,
        'runtime': time.time() - start
    }

    # 4. OPTIMAL ALLOCATION (coarse grid search)
    start = time.time()

    best_obj_optimal = 0
    best_spent = None

    # Try only a few key allocations (much faster)
    test_allocations = [
        (0.5, 0.3, 0.2),  # Search-heavy
        (0.4, 0.4, 0.2),  # Balanced
        (0.3, 0.5, 0.2),  # Display-heavy
        (0.4, 0.3, 0.3),  # More video
        (0.45, 0.35, 0.2),  # Refined search
    ]

    for w1, w2, w3 in test_allocations:
        budget_allocation = {
            'Search': B * w1,
            'Display': B * w2,
            'Video': B * w3
        }

        total_obj = 0
        actual_spent = {}

        for pname in platform_names:
            x, obj, spent = solve_platform_allocation(
                platforms[pname]['causal_score'],
                platforms[pname]['cost'],
                budget_allocation[pname]
            )
            total_obj += obj
            actual_spent[pname] = spent

        if total_obj > best_obj_optimal:
            best_obj_optimal = total_obj
            best_spent = actual_spent.copy()

    iroas_optimal = best_obj_optimal / sum(best_spent.values()) if sum(best_spent.values()) > 0 else 0

    results['Optimal'] = {
        'objective': best_obj_optimal,
        'iroas': iroas_optimal,
        'budget_split': best_spent,
        'runtime': time.time() - start
    }

    return results


if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("SECTION 6: CROSS-PLATFORM BUDGET ALLOCATION")
    print("="*80)

    # Run multiple times
    n_runs = 10
    all_results = []

    for seed in range(42, 42 + n_runs):
        all_results.append(run_simulation(seed))

    # Aggregate results
    methods = ['Single Platform', 'Equal Split', 'Proportional', 'Optimal']

    summary = []
    for method in methods:
        objectives = [r[method]['objective'] for r in all_results]
        iroas_vals = [r[method]['iroas'] for r in all_results]

        row = {
            'Strategy': method,
            'Objective Value': f"${np.mean(objectives):.2f}",
            'iROAS': f"{np.mean(iroas_vals):.2f}×"
        }

        # Add budget splits
        if method == 'Single Platform':
            best_platforms = [r[method]['best_platform'] for r in all_results]
            from collections import Counter
            most_common = Counter(best_platforms).most_common(1)[0]
            row['Note'] = f"{most_common[0]} ({most_common[1]}/{n_runs} runs)"
        else:
            # Average budget split
            search_budgets = [r[method]['budget_split']['Search'] for r in all_results]
            display_budgets = [r[method]['budget_split']['Display'] for r in all_results]
            video_budgets = [r[method]['budget_split']['Video'] for r in all_results]

            row['Search %'] = f"{np.mean(search_budgets)/20*100:.0f}%"
            row['Display %'] = f"{np.mean(display_budgets)/20*100:.0f}%"
            row['Video %'] = f"{np.mean(video_budgets)/20*100:.0f}%"

        summary.append(row)

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    # Calculate improvement
    baseline_iroas = np.mean([r['Single Platform']['iroas'] for r in all_results])
    optimal_iroas = np.mean([r['Optimal']['iroas'] for r in all_results])
    improvement = (optimal_iroas - baseline_iroas) / baseline_iroas * 100

    print(f"\nImprovement (Optimal vs Single Platform): {improvement:+.1f}%")
    print(f"Average runtime (Optimal): {np.mean([r['Optimal']['runtime'] for r in all_results]):.2f}s")
