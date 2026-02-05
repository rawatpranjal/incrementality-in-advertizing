"""
Causal Primitives Simulation for Autobidding Selection Bias

Demonstrates value destruction from using observational pCVR vs causal iCVR
by modeling the full causal structure with primitives (U, Y(0), Y(1), D).

Key approach:
- 2 market goods represent observable segments (U=0: low-intent, U=1: high-intent)
- Oracle system uses iCVR (incremental value) for valuations
- Naive system uses pCVR (observational value) for valuations
- Each system reaches its own FPPE equilibrium
- Value destruction measured by evaluating naive's allocation at TRUE values
"""

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import sys
sys.path.append('outputs')

from scenarios_config import SCENARIOS, validate_scenario
from fppe_with_budget_constraint import compute_fppe_with_explicit_budget


def calculate_population_metrics(primitives):
    """
    Calculates population-level causal metrics from primitives.
    """
    p_u = primitives['p_u']

    # Segment-level lifts
    iCVR_u1 = primitives['p_y1_u1'] - primitives['p_y0_u1']
    iCVR_u0 = primitives['p_y1_u0'] - primitives['p_y0_u0']

    # Population Average Treatment Effect (ATE)
    iCVR_avg = iCVR_u1 * p_u + iCVR_u0 * (1 - p_u)

    # Population organic conversion rate
    oCVR = primitives['p_y0_u1'] * p_u + primitives['p_y0_u0'] * (1 - p_u)

    # Population selection rate
    p_d = primitives['p_d_u1'] * p_u + primitives['p_d_u0'] * (1 - p_u)

    # Population pCVR (what naive system observes)
    if p_d > 1e-6:
        pCVR = (primitives['p_y1_u1'] * primitives['p_d_u1'] * p_u +
                primitives['p_y1_u0'] * primitives['p_d_u0'] * (1 - p_u)) / p_d
    else:
        pCVR = np.nan
        warnings.warn("Selection rate p_d too low, pCVR undefined")

    # Observational error
    obs_error = pCVR - iCVR_avg if not np.isnan(pCVR) else np.nan

    # Selection bias magnitude
    selection_bias_magnitude = primitives['p_d_u1'] - primitives['p_d_u0']
    selection_ratio = primitives['p_d_u1'] / primitives['p_d_u0'] if primitives['p_d_u0'] > 1e-6 else np.inf

    # Correlation direction
    correlation_sign = "positive" if iCVR_u1 > iCVR_u0 else "negative" if iCVR_u1 < iCVR_u0 else "zero"

    return {
        # Segment-level
        'iCVR_u1': iCVR_u1,
        'iCVR_u0': iCVR_u0,
        'oCVR_u1': primitives['p_y0_u1'],
        'oCVR_u0': primitives['p_y0_u0'],
        'pCVR_u1': primitives['p_y1_u1'],
        'pCVR_u0': primitives['p_y1_u0'],

        # Population-level
        'iCVR_avg': iCVR_avg,
        'oCVR': oCVR,
        'pCVR': pCVR,
        'obs_error': obs_error,
        'p_d': p_d,

        # Bias characterization
        'selection_bias_magnitude': selection_bias_magnitude,
        'selection_ratio': selection_ratio,
        'correlation_sign': correlation_sign,
    }


def construct_segment_valuations(primitives, num_bidders, vpc_range, seed):
    """
    Constructs valuation matrices for oracle and naive systems.

    Market structure: 2 goods (segments)
    - Good 0: Low-intent segment (U=0)
    - Good 1: High-intent segment (U=1)

    Oracle valuations: Based on incremental lift (iCVR)
    Naive valuations: Based on observational conversion (pCVR = Y(1))
    """
    np.random.seed(seed)

    # Generate per-bidder value-per-conversion (VPC)
    vpc = np.random.uniform(vpc_range[0], vpc_range[1], num_bidders)

    # Calculate segment-level metrics
    metrics = calculate_population_metrics(primitives)

    # Oracle valuations (TRUE incremental value)
    V_oracle = np.zeros((num_bidders, 2))
    V_oracle[:, 0] = metrics['iCVR_u0'] * vpc  # Low-intent segment
    V_oracle[:, 1] = metrics['iCVR_u1'] * vpc  # High-intent segment

    # Naive valuations (OBSERVATIONAL value)
    V_naive = np.zeros((num_bidders, 2))
    V_naive[:, 0] = primitives['p_y1_u0'] * vpc  # Low-intent observational
    V_naive[:, 1] = primitives['p_y1_u1'] * vpc  # High-intent observational

    return V_oracle, V_naive, vpc, metrics


def run_single_scenario(scenario_name, scenario_primitives, market_config, seed, verbose=False):
    """
    Runs FPPE comparison for a single scenario.

    Each system reaches its own equilibrium using FPPE, then we evaluate
    the naive system's allocation using TRUE values to measure value destruction.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running scenario: {scenario_name}")
        print(f"Seed: {seed}")
        print(f"{'='*80}\n")

    # Validate primitives
    is_valid, msg = validate_scenario(scenario_primitives)
    if not is_valid:
        return {"status": "invalid", "error": msg}

    # Construct valuations
    V_oracle, V_naive, vpc, metrics = construct_segment_valuations(
        scenario_primitives,
        market_config['num_bidders'],
        market_config['vpc_range'],
        seed
    )

    # Check for zero-lift scenarios (no value to capture)
    if np.max(V_oracle) < 1e-6:
        # All lifts are zero - no incremental value
        return {
            'scenario_name': scenario_name,
            'seed': seed,
            'status': 'success',
            **scenario_primitives,
            **metrics,
            'value_oracle': 0.0,
            'value_naive_reported': np.sum(V_naive) * market_config['budget_multiplier'],
            'true_value_naive': 0.0,
            'gap': 0.0,
            'lift_loss_pct': 100.0,  # 100% loss if oracle value is zero
            'segment_alloc_obs_low': 0.0,
            'segment_alloc_obs_high': 0.0,
            'segment_alloc_causal_low': 0.0,
            'segment_alloc_causal_high': 0.0,
            'ratio_obs': 0.0,
            'ratio_causal': 0.0,
            'mean_lambda_obs': 0.0,
            'mean_lambda_causal': 0.0,
        }

    # Calculate budgets (neutral baseline)
    avg_value_per_bidder = (np.sum(V_oracle, axis=1) + np.sum(V_naive, axis=1)) / 2
    budgets = avg_value_per_bidder * market_config['budget_multiplier']

    if verbose:
        print(f"Market configuration:")
        print(f"  Bidders: {market_config['num_bidders']}")
        print(f"  Total budget: ${np.sum(budgets):.2f}")
        print(f"  Budget per bidder: ${np.mean(budgets):.2f}")
        print()

    # Solve oracle system with oracle valuations
    result_oracle = compute_fppe_with_explicit_budget(V_oracle, budgets, verbose=False)

    if result_oracle['status'] not in ['optimal', 'optimal_inaccurate']:
        return {"status": f"solver_failed_oracle_{result_oracle['status']}", "scenario_name": scenario_name, "seed": seed}

    # Solve naive system with naive valuations
    result_naive = compute_fppe_with_explicit_budget(V_naive, budgets, verbose=False)

    if result_naive['status'] not in ['optimal', 'optimal_inaccurate']:
        return {"status": f"solver_failed_naive_{result_naive['status']}", "scenario_name": scenario_name, "seed": seed}

    # CRITICAL: Evaluate naive's allocation using TRUE values
    true_value_naive = np.sum(result_naive['allocations'] * V_oracle)
    value_oracle = result_oracle['total_revenue']

    # Performance gap
    gap = value_oracle - true_value_naive
    lift_loss_pct = (gap / value_oracle * 100) if value_oracle > 1e-6 else 0

    # Pacing analysis
    lambda_obs = result_naive['pacing_multipliers']
    lambda_causal = result_oracle['pacing_multipliers']

    # Allocation analysis
    alloc_obs = result_naive['allocations']
    alloc_causal = result_oracle['allocations']

    # Segment allocation (which segment gets more?)
    segment_allocation_obs = np.sum(alloc_obs, axis=0)  # [low, high]
    segment_allocation_causal = np.sum(alloc_causal, axis=0)

    # Allocation ratio (high-intent / low-intent)
    ratio_obs = segment_allocation_obs[1] / (segment_allocation_obs[0] + 1e-9)
    ratio_causal = segment_allocation_causal[1] / (segment_allocation_causal[0] + 1e-9)

    if verbose:
        print(f"ORACLE SYSTEM (causal iCVR):")
        print(f"  Value: ${value_oracle:.2f}")
        print(f"  Allocation: Low={segment_allocation_causal[0]:.2f}, High={segment_allocation_causal[1]:.2f}")
        print(f"  Ratio (high/low): {ratio_causal:.2f}x")
        print(f"  Mean λ: {np.mean(lambda_causal):.4f}")
        print()

        print(f"NAIVE SYSTEM (observational pCVR):")
        print(f"  Reported value (using V_naive): ${result_naive['total_revenue']:.2f}")
        print(f"  TRUE value (using V_oracle): ${true_value_naive:.2f}")
        print(f"  Allocation: Low={segment_allocation_obs[0]:.2f}, High={segment_allocation_obs[1]:.2f}")
        print(f"  Ratio (high/low): {ratio_obs:.2f}x")
        print(f"  Mean λ: {np.mean(lambda_obs):.4f}")
        print()

        print(f"VALUE DESTRUCTION:")
        print(f"  Gap: ${gap:.2f}")
        print(f"  Lift loss: {lift_loss_pct:.2f}%")
        print()

    # Return full result
    return {
        # Identifiers
        'scenario_name': scenario_name,
        'seed': seed,
        'status': 'success',

        # Primitives
        **scenario_primitives,

        # Population metrics
        **metrics,

        # Market outcomes
        'value_oracle': value_oracle,
        'value_naive_reported': result_naive['total_revenue'],
        'true_value_naive': true_value_naive,
        'gap': gap,
        'lift_loss_pct': lift_loss_pct,

        # Allocation
        'segment_alloc_obs_low': segment_allocation_obs[0],
        'segment_alloc_obs_high': segment_allocation_obs[1],
        'segment_alloc_causal_low': segment_allocation_causal[0],
        'segment_alloc_causal_high': segment_allocation_causal[1],
        'ratio_obs': ratio_obs,
        'ratio_causal': ratio_causal,

        # Pacing
        'mean_lambda_obs': np.mean(lambda_obs),
        'mean_lambda_causal': np.mean(lambda_causal),
    }


def run_full_simulation(scenarios, market_config, num_iterations=50, verbose=False):
    """
    Runs full simulation across all scenarios with multiple iterations.
    """
    all_results = []

    for scenario_key, scenario_data in tqdm(scenarios.items(), desc="Scenarios"):
        scenario_name = scenario_data['name']

        for iteration in range(num_iterations):
            seed = 1000 + iteration

            result = run_single_scenario(
                scenario_name,
                scenario_data,
                market_config,
                seed,
                verbose=False
            )

            if result['status'] == 'success':
                all_results.append(result)
            else:
                warnings.warn(f"Iteration {iteration} failed for {scenario_name}: {result.get('status', 'unknown')}")

    return pd.DataFrame(all_results)


def analyze_results(df, output_path=None):
    """
    Analyzes simulation results and generates summary statistics.
    """
    # Group by scenario
    summary = df.groupby('scenario_name').agg({
        'lift_loss_pct': ['mean', 'std', 'min', 'max'],
        'value_oracle': 'mean',
        'true_value_naive': 'mean',
        'gap': 'mean',
        'obs_error': 'mean',
        'ratio_obs': 'mean',
        'ratio_causal': 'mean',
        'mean_lambda_obs': 'mean',
        'mean_lambda_causal': 'mean',
        'selection_ratio': 'first',
        'correlation_sign': 'first',
    }).round(4)

    # Build output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CAUSAL PRIMITIVES SIMULATION RESULTS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("This simulation demonstrates value destruction from using observational")
    output_lines.append("pCVR instead of causal iCVR in autobidding allocation.")
    output_lines.append("")
    output_lines.append("CAUSAL STRUCTURE:")
    output_lines.append("- U: Unobserved intent (0=low, 1=high)")
    output_lines.append("- Y(0): Organic conversion (without ad)")
    output_lines.append("- Y(1): Treated conversion (with ad)")
    output_lines.append("- D: Selection (whether user sees/clicks ad)")
    output_lines.append("")
    output_lines.append("CORRELATION TYPES:")
    output_lines.append("- Positive: high-intent users have BOTH high organic AND high lift")
    output_lines.append("- Negative: high-intent users have high organic BUT low lift (sure things)")
    output_lines.append("- Zero: no correlation between intent and lift")
    output_lines.append("")
    output_lines.append("SELECTION BIAS:")
    output_lines.append("- Selection ratio: P(D=1|U=1) / P(D=1|U=0)")
    output_lines.append("- Measures how much more likely high-intent users are selected")
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("SCENARIO COMPARISON")
    output_lines.append("-" * 80)
    output_lines.append(summary.to_string())
    output_lines.append("")
    output_lines.append("")

    # Sort by lift loss and provide insights
    output_lines.append("KEY FINDINGS (sorted by value destruction)")
    output_lines.append("-" * 80)
    output_lines.append("")

    sorted_scenarios = summary.sort_values(('lift_loss_pct', 'mean'), ascending=False)

    for scenario_name in sorted_scenarios.index:
        row = summary.loc[scenario_name]
        mean_loss = row[('lift_loss_pct', 'mean')]
        std_loss = row[('lift_loss_pct', 'std')]
        selection_ratio = row[('selection_ratio', 'first')]
        correlation = row[('correlation_sign', 'first')]
        ratio_obs = row[('ratio_obs', 'mean')]
        ratio_causal = row[('ratio_causal', 'mean')]
        obs_error = row[('obs_error', 'mean')]

        output_lines.append(f"{scenario_name}:")
        output_lines.append(f"  Lift loss: {mean_loss:.2f}% ± {std_loss:.2f}%")
        output_lines.append(f"  Selection ratio: {selection_ratio:.2f}x")
        output_lines.append(f"  Correlation: {correlation}")
        output_lines.append(f"  Observational error: {obs_error:.4f}")
        output_lines.append(f"  Allocation ratio: naive={ratio_obs:.2f}x vs oracle={ratio_causal:.2f}x")
        output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("INTERPRETATION:")
    output_lines.append("")
    output_lines.append("VALUE DESTRUCTION MECHANISMS:")
    output_lines.append("")
    output_lines.append("1. NEGATIVE CORRELATION (Sure Things vs Persuadables):")
    output_lines.append("   - Naive system targets high-intent users (high organic, high pCVR)")
    output_lines.append("   - But these are 'sure things' with LOW incremental lift")
    output_lines.append("   - Misses 'persuadables' (low organic, high lift)")
    output_lines.append("   - Paradox: Strongest selection bias can cause LESS damage if it")
    output_lines.append("     accidentally selects the right (high-lift) segment")
    output_lines.append("")
    output_lines.append("2. POSITIVE CORRELATION:")
    output_lines.append("   - Naive system targets high-intent users")
    output_lines.append("   - These ALSO have high lift (lucky coincidence)")
    output_lines.append("   - Still causes damage because allocation is WRONG, not just direction")
    output_lines.append("   - Observational error inflates valuations → over-allocation")
    output_lines.append("")
    output_lines.append("3. ZERO CORRELATION (Control):")
    output_lines.append("   - Equal lift across segments")
    output_lines.append("   - Oracle is indifferent to allocation")
    output_lines.append("   - Any misallocation causes value loss")
    output_lines.append("")
    output_lines.append("=" * 80)

    output_text = "\n".join(output_lines)

    # Print to console
    print(output_text)

    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
        print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    print("CAUSAL PRIMITIVES SIMULATION")
    print("=" * 80)
    print()

    # Market configuration
    market_config = {
        'num_bidders': 15,
        'vpc_range': (50, 150),
        'budget_multiplier': 0.25,
    }

    print("Market configuration:")
    for key, val in market_config.items():
        print(f"  {key}: {val}")
    print()

    # Run simulation
    print("=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    print()

    results_df = run_full_simulation(
        scenarios=SCENARIOS,
        market_config=market_config,
        num_iterations=100,
        verbose=False
    )

    # Save detailed results
    results_df.to_csv('causal_primitives_detailed.csv', index=False)
    print(f"\nDetailed results saved to: causal_primitives_detailed.csv")
    print()

    # Analyze and output
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    summary = analyze_results(results_df, output_path='causal_primitives_results.txt')

    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
