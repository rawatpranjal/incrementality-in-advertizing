"""
Section 5: Frequency Capping
Models diminishing returns from repeated exposures
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

VALUE_PER_CONVERSION = 100.0


def compute_diminishing_value(tau, k, alpha=0.6):
    """
    Compute value of k-th exposure with diminishing returns
    tau: base treatment effect
    k: exposure number (1, 2, 3, ...)
    alpha: diminishing returns parameter (higher = faster decay)
    """
    return tau / (1 + alpha * (k - 1))


def solve_frequency_capped(causal_score, c, B, F_max, M, alpha=0.6):
    """
    Solve MILP with frequency capping

    Decision variables: x[j,k] = 1 if user j gets k exposures

    Returns: exposure counts per user, objective value
    """
    # For each user, decide how many exposures (0 to F_max)
    # We'll use a greedy approximation for efficiency

    # Compute value per exposure for each user at each frequency
    user_values = []
    for j in range(M):
        user_freq_values = []
        for k in range(1, F_max + 1):
            marginal_value = compute_diminishing_value(causal_score[j], k, alpha)
            # Cost increases with frequency (retargeting costs more)
            marginal_cost = c[j] * (1 + 0.15 * (k - 1))
            user_freq_values.append({
                'user': j,
                'freq': k,
                'value': marginal_value,
                'cost': marginal_cost,
                'ratio': marginal_value / marginal_cost if marginal_cost > 0 else 0
            })
        user_values.extend(user_freq_values)

    # Greedy allocation by bang-for-buck ratio
    user_values.sort(key=lambda x: x['ratio'], reverse=True)

    exposures = np.zeros(M)
    budget_used = 0
    total_value = 0

    for item in user_values:
        j = item['user']
        k = item['freq']
        if exposures[j] < k and budget_used + item['cost'] <= B:
            exposures[j] = k
            budget_used += item['cost']
            total_value += item['value']

    return exposures, total_value, budget_used


def run_simulation(seed=42):
    """
    Simulate frequency capping strategies
    Returns: dict with results for each frequency cap
    """
    np.random.seed(seed)

    # Data generation
    M = 500  # Fewer users means more frequency per user
    B = 20.0
    alpha = 0.6  # Diminishing returns parameter

    p_0 = np.random.beta(2, 8, M)
    epsilon = np.random.normal(0, 0.005, M)
    tau = 0.08 / (1 + 10 * p_0) + epsilon
    tau = np.clip(tau, 0.001, 0.15)
    ctr = np.random.beta(3, 20, M)
    cpc = np.random.exponential(0.3, M) + 0.05  # Lower cost = more frequency
    c = ctr * cpc
    v = VALUE_PER_CONVERSION

    # Base causal scores (single exposure)
    causal_score = ctr * tau * v

    results = {}

    # 1. NO CAP (naive: doesn't model diminishing returns)
    start = time.time()
    # Naive system uses first-impression value for all frequency decisions
    # This is common in practice when systems don't model frequency response curves

    first_imp_value = causal_score  # Use first impression value
    first_imp_cost = c
    first_imp_ratio = first_imp_value / first_imp_cost
    sorted_idx = np.argsort(-first_imp_ratio)

    exposures_nocap = np.zeros(M)
    budget_used_nocap = 0
    obj_nocap = 0

    # Allocate based on first-impression value (ignoring diminishing returns)
    for idx in sorted_idx:
        for k in range(1, 11):
            # Naive system: uses first impression value for decision
            perceived_value = first_imp_value[idx]
            actual_cost = c[idx] * (1 + 0.15 * (k - 1))

            # Decision based on perceived value
            if budget_used_nocap + actual_cost <= B and perceived_value > actual_cost:
                exposures_nocap[idx] = k
                budget_used_nocap += actual_cost

                # But actual value received has diminishing returns
                actual_value = compute_diminishing_value(causal_score[idx], k, alpha)
                obj_nocap += actual_value
            else:
                break

        if budget_used_nocap >= B * 0.98:
            break

    iroas_nocap = obj_nocap / budget_used_nocap if budget_used_nocap > 0 else 0
    avg_freq_nocap = exposures_nocap[exposures_nocap > 0].mean() if (exposures_nocap > 0).sum() > 0 else 0

    results['No Cap'] = {
        'objective': obj_nocap,
        'budget_used': budget_used_nocap,
        'iroas': iroas_nocap,
        'avg_frequency': avg_freq_nocap,
        'users_reached': (exposures_nocap > 0).sum(),
        'runtime': time.time() - start
    }

    # 2. CAP = 3
    start = time.time()
    exposures_cap3, obj_cap3, budget_cap3 = solve_frequency_capped(
        causal_score, c, B, F_max=3, M=M, alpha=alpha
    )
    iroas_cap3 = obj_cap3 / budget_cap3 if budget_cap3 > 0 else 0
    avg_freq_cap3 = exposures_cap3[exposures_cap3 > 0].mean() if (exposures_cap3 > 0).sum() > 0 else 0

    results['Cap = 3'] = {
        'objective': obj_cap3,
        'budget_used': budget_cap3,
        'iroas': iroas_cap3,
        'avg_frequency': avg_freq_cap3,
        'users_reached': (exposures_cap3 > 0).sum(),
        'runtime': time.time() - start
    }

    # 3. CAP = 5
    start = time.time()
    exposures_cap5, obj_cap5, budget_cap5 = solve_frequency_capped(
        causal_score, c, B, F_max=5, M=M, alpha=alpha
    )
    iroas_cap5 = obj_cap5 / budget_cap5 if budget_cap5 > 0 else 0
    avg_freq_cap5 = exposures_cap5[exposures_cap5 > 0].mean() if (exposures_cap5 > 0).sum() > 0 else 0

    results['Cap = 5'] = {
        'objective': obj_cap5,
        'budget_used': budget_cap5,
        'iroas': iroas_cap5,
        'avg_frequency': avg_freq_cap5,
        'users_reached': (exposures_cap5 > 0).sum(),
        'runtime': time.time() - start
    }

    # 4. OPTIMAL CAP (search over cap values)
    start = time.time()
    best_iroas = 0
    best_cap = 1
    best_results = None

    for F in range(1, 11):
        exposures_f, obj_f, budget_f = solve_frequency_capped(
            causal_score, c, B, F_max=F, M=M, alpha=alpha
        )
        iroas_f = obj_f / budget_f if budget_f > 0 else 0
        if iroas_f > best_iroas:
            best_iroas = iroas_f
            best_cap = F
            best_results = {
                'objective': obj_f,
                'budget_used': budget_f,
                'iroas': iroas_f,
                'avg_frequency': exposures_f[exposures_f > 0].mean() if (exposures_f > 0).sum() > 0 else 0,
                'users_reached': (exposures_f > 0).sum()
            }

    best_results['runtime'] = time.time() - start
    best_results['optimal_cap'] = best_cap

    results['Optimal Cap'] = best_results

    return results


if __name__ == "__main__":
    import pandas as pd

    print("="*80)
    print("SECTION 5: FREQUENCY CAPPING")
    print("="*80)

    # Run multiple times
    n_runs = 10
    all_results = []

    for seed in range(42, 42 + n_runs):
        all_results.append(run_simulation(seed))

    # Aggregate results
    methods = ['No Cap', 'Cap = 3', 'Cap = 5', 'Optimal Cap']
    method_labels = {
        'No Cap': 'No Cap (Naive)',
        'Cap = 3': 'Hard Cap = 3',
        'Cap = 5': 'Hard Cap = 5',
        'Optimal Cap': 'Optimal Cap'
    }

    summary = []
    for method in methods:
        objectives = [r[method]['objective'] for r in all_results]
        iroas_vals = [r[method]['iroas'] for r in all_results]
        avg_freqs = [r[method]['avg_frequency'] for r in all_results]
        users = [r[method]['users_reached'] for r in all_results]

        row = {
            'Strategy': method_labels[method],
            'Objective Value': f"${np.mean(objectives):.2f}",
            'iROAS': f"{np.mean(iroas_vals):.2f}×",
            'Avg Frequency': f"{np.mean(avg_freqs):.2f}",
            'Users Reached': f"{int(np.mean(users))}"
        }

        if method == 'Optimal Cap':
            optimal_caps = [r[method]['optimal_cap'] for r in all_results]
            row['Optimal Cap'] = f"{np.mean(optimal_caps):.1f}"

        summary.append(row)

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    # Calculate improvement
    baseline_iroas = np.mean([r['No Cap']['iroas'] for r in all_results])
    optimal_iroas = np.mean([r['Optimal Cap']['iroas'] for r in all_results])
    improvement = (optimal_iroas - baseline_iroas) / baseline_iroas * 100

    print(f"\nImprovement (Optimal Cap vs Naive No Cap): {improvement:+.1f}%")
    print(f"\nScenario: Naive system uses first-impression value for all frequency decisions")
    print(f"          (doesn't model diminishing returns, leading to over-frequency)")
    print(f"\nDiminishing returns: τ_k = τ_1 / (1 + 0.6*(k-1))")
    print(f"Cost increases: c_k = c_1 * (1 + 0.15*(k-1))")
