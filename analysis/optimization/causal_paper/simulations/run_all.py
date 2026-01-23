"""
Run all simulations for Causal Optimization Paper
Executes all section simulations and generates combined results
"""

import sys
import time
import os
from io import StringIO
import subprocess


def run_simulation(script_name):
    """Run a simulation script and capture output"""
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )
    return result.stdout


def main():
    print("="*80)
    print("CAUSAL OPTIMIZATION IN ADVERTISING: COMPREHENSIVE SIMULATION RESULTS")
    print("="*80)
    print(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    simulations = [
        ('sim_basic.py', 'Section 2: Standard vs Lift-Based Optimization'),
        ('sim_fractional.py', 'Section 3: Fractional Relaxation'),
        ('sim_pacing.py', 'Section 4: Multi-Period Budget Pacing'),
        ('sim_frequency.py', 'Section 5: Frequency Capping'),
        ('sim_crossplatform.py', 'Section 6: Cross-Platform Budget Allocation')
    ]

    all_outputs = []

    for i, (script, description) in enumerate(simulations, 1):
        print(f"[{i}/{len(simulations)}] Running {description}...")
        start = time.time()
        output = run_simulation(script)
        duration = time.time() - start
        all_outputs.append((description, output))
        print(f"      Completed in {duration:.2f}s")

    print()
    print("="*80)
    print("ALL SIMULATIONS COMPLETED")
    print("="*80)

    # Display all results
    print("\n\n")
    for description, output in all_outputs:
        print(output)
        print("\n")

    # Create results directory if it doesn't exist
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    # Write to file
    output_file = os.path.join(results_dir, "all_simulations.txt")
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CAUSAL OPTIMIZATION IN ADVERTISING: COMPREHENSIVE SIMULATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for description, output in all_outputs:
            f.write(f"\n{'='*80}\n")
            f.write(f"{description.upper()}\n")
            f.write(f"{'='*80}\n\n")
            f.write(output)
            f.write("\n\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
