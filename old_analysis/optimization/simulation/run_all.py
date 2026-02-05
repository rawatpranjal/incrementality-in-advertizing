"""
Run all simulations and save outputs to log file
No plots displayed, all stdout captured
"""

import sys
import os
from io import StringIO
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no display

# Capture all output
log_buffer = StringIO()

class DualWriter:
    def __init__(self, file_obj, terminal):
        self.file = file_obj
        self.terminal = terminal

    def write(self, message):
        self.file.write(message)
        self.terminal.write(message)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

# Redirect stdout
original_stdout = sys.stdout
sys.stdout = DualWriter(log_buffer, original_stdout)

print(f"Simulation started at {datetime.now()}")
print("="*80)

# Import and run simulations
print("\n1. BIDDING SIMULATION")
print("-"*40)
import bidding_simulation
bidding_results = bidding_simulation.main()

print("\n2. SLATE RANKING SIMULATION")
print("-"*40)
import slate_simulation
slate_results, slate_summary = slate_simulation.main()

print("\n3. GENERATING VISUALIZATIONS")
print("-"*40)
import visualization
visualization.main()

print("\n" + "="*80)
print(f"Simulation completed at {datetime.now()}")

# Restore stdout
sys.stdout = original_stdout

# Save log to file
log_content = log_buffer.getvalue()
with open('/Users/pranjal/Code/marketplace-incrementality/optimization/simulation/simulation_log.txt', 'w') as f:
    f.write(log_content)

print("\nLog saved to simulation_log.txt")
print("All plots saved to results/")
print("No plots displayed.")