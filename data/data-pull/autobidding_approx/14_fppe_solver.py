"""
FPPE Solver: Find optimal pacing multipliers for budget-constrained bidders.

Algorithm: Iterative best-response with first-price auctions
- Initialize λ[i] for all campaigns
- Iterate:
  1. Run auctions with bids = λ[i] × V[i,j]
  2. Compute spending for each campaign
  3. Update λ[i] to respect budget constraints
- Converge when λ changes are small

Input: market_*.npz (B, V, lambda_obs)
Output: fppe_solution_*.npz (lambda_optimal, allocations, bids, prices)
"""

import numpy as np
import argparse
from tqdm import tqdm

print("="*80)
print("FPPE SOLVER: OPTIMAL PACING MULTIPLIERS")
print("="*80)
print()

# Parse arguments
parser = argparse.ArgumentParser(description='Solve FPPE for optimal pacing')
parser.add_argument('--market', type=str, required=True, help='Market file (*.npz)')
parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations')
parser.add_argument('--tol', type=float, default=1e-4, help='Convergence tolerance')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Lambda update rate')
parser.add_argument('--reserve_price', type=float, default=0.0, help='Reserve price for auctions')
parser.add_argument('--slots_per_auction', type=int, default=1, help='Number of winners per auction (K in top-K allocation)')
args = parser.parse_args()

# Load market
print(f"Loading market from {args.market}...")
data = np.load(args.market, allow_pickle=True)

B = data['B']  # budgets (N,)
V = data['V']  # valuations (N, M)
lambda_obs = data['lambda_obs']  # observed pacing (N,)
date = str(data['date'])

N, M = V.shape
print(f"Market size: {N} campaigns × {M} auctions")
print(f"Date: {date}")
print()

print("Market statistics:")
print(f"  Budgets: mean=${B.mean():.2f}, median=${np.median(B):.2f}")
print(f"  Values: mean=${V[V>0].mean():.4f}, median=${np.median(V[V>0]):.4f}")
print(f"  Observed pacing: mean={lambda_obs.mean():.4f}, median={np.median(lambda_obs):.4f}")
print()

# Initialize lambda
lambda_current = np.ones(N)  # Start with no pacing
spend = np.zeros(N)
allocations = np.zeros((N, M))

print("="*80)
print("RUNNING ITERATIVE BEST-RESPONSE")
print("="*80)
print()

print(f"Max iterations: {args.max_iter}")
print(f"Convergence tolerance: {args.tol}")
print(f"Slots per auction: {args.slots_per_auction}")
print()

converged = False
iteration = 0

for iteration in tqdm(range(args.max_iter), desc="FPPE iterations"):

    # Run first-price auctions
    bids = lambda_current[:, np.newaxis] * V  # (N, M)

    # For each auction, find winner
    allocations_new = np.zeros((N, M))
    prices = np.zeros(M)

    for j in range(M):
        auction_bids = bids[:, j]

        # Only consider positive bids
        valid_bidders = auction_bids > args.reserve_price

        if valid_bidders.any():
            # Find top-K bidders (K = slots_per_auction)
            K = min(args.slots_per_auction, valid_bidders.sum())

            # Get top K bidders
            top_k_indices = np.argsort(auction_bids)[-K:]

            # Allocate to all top-K winners
            for winner_idx in top_k_indices:
                if auction_bids[winner_idx] > args.reserve_price:
                    allocations_new[winner_idx, j] = 1.0

            # Price: each winner pays their own bid (first-price)
            # For budget tracking, use average winning bid as "price"
            prices[j] = auction_bids[top_k_indices].mean()

    # Calculate spending: each campaign pays their own bid (first-price)
    spend_new = (allocations_new * bids).sum(axis=1)

    # Update lambda to satisfy budget constraints
    lambda_new = lambda_current.copy()

    for i in range(N):
        if B[i] > 1e-6:  # Has meaningful budget
            # Target: spend ≈ budget
            if spend_new[i] > B[i]:
                # Overspending: reduce lambda
                lambda_new[i] = lambda_current[i] * (B[i] / (spend_new[i] + 1e-9))
            elif spend_new[i] < B[i] * 0.95 and lambda_current[i] < 1.0:
                # Underspending and paced: increase lambda
                increase_factor = min(1.5, B[i] / (spend_new[i] + 1e-9))
                lambda_new[i] = min(1.0, lambda_current[i] * increase_factor)
        else:
            # No budget: set lambda to 0
            lambda_new[i] = 0.0

    # Smooth update
    lambda_new = (1 - args.learning_rate) * lambda_current + args.learning_rate * lambda_new
    lambda_new = np.clip(lambda_new, 0.0, 1.0)

    # Check convergence
    lambda_change = np.abs(lambda_new - lambda_current).max()

    if lambda_change < args.tol:
        converged = True
        iteration += 1
        break

    lambda_current = lambda_new
    allocations = allocations_new
    spend = spend_new

print()
if converged:
    print(f"✓ Converged after {iteration} iterations")
else:
    print(f"⚠️  Reached max iterations ({args.max_iter}) without full convergence")

print()

# Final solution
lambda_optimal = lambda_current
final_bids = lambda_optimal[:, np.newaxis] * V
final_spend = spend

print("="*80)
print("OPTIMAL SOLUTION")
print("="*80)
print()

print("Optimal pacing multipliers:")
print(f"  Mean: {lambda_optimal.mean():.4f}")
print(f"  Median: {np.median(lambda_optimal):.4f}")
print(f"  p10-p90: {np.quantile(lambda_optimal, 0.10):.4f} - {np.quantile(lambda_optimal, 0.90):.4f}")
print(f"  % >= 0.95: {(lambda_optimal >= 0.95).mean()*100:.1f}%")
print()

print("Budget utilization:")
budget_utilization = final_spend / (B + 1e-9)
budget_utilization = np.clip(budget_utilization, 0, 2)  # Cap for display
print(f"  Mean: {budget_utilization.mean():.4f}")
print(f"  Median: {np.median(budget_utilization):.4f}")
print(f"  % using >95% of budget: {(budget_utilization >= 0.95).mean()*100:.1f}%")
print()

print("Allocations:")
print(f"  Total auctions won: {allocations.sum():.0f} / {M}")
print(f"  Avg auctions per campaign: {allocations.sum(axis=1).mean():.2f}")
print(f"  Campaigns with >0 wins: {(allocations.sum(axis=1) > 0).sum()} / {N}")
print()

print("Equilibrium bids:")
winning_bids = final_bids[allocations > 0]
print(f"  Mean winning bid: ${winning_bids.mean():.4f}")
print(f"  Median winning bid: ${np.median(winning_bids):.4f}")
print()

# Compare to observed
print("="*80)
print("COMPARISON TO OBSERVED")
print("="*80)
print()

corr = np.corrcoef(lambda_optimal, lambda_obs)[0, 1]
rmse = np.sqrt(((lambda_optimal - lambda_obs)**2).mean())
mae = np.abs(lambda_optimal - lambda_obs).mean()

print(f"Correlation (λ_optimal, λ_obs): {corr:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print()

print("Pacing distribution comparison:")
print(f"  Observed:  mean={lambda_obs.mean():.4f}, %>=0.95={(lambda_obs>=0.95).mean()*100:.1f}%")
print(f"  Optimal:   mean={lambda_optimal.mean():.4f}, %>=0.95={(lambda_optimal>=0.95).mean()*100:.1f}%")
print()

# Save solution
output_file = args.market.replace('market_', 'fppe_solution_')
print(f"Saving solution to {output_file}...")

np.savez(output_file,
         lambda_optimal=lambda_optimal,
         lambda_obs=lambda_obs,
         allocations=allocations,
         equilibrium_bids=final_bids,
         prices=prices,
         spend=final_spend,
         budget_utilization=budget_utilization,
         converged=converged,
         iterations=iteration,
         correlation=corr,
         rmse=rmse,
         mae=mae,
         date=date)

print(f"✓ Saved {output_file}")
print()

print("="*80)
print("FPPE SOLUTION COMPLETE")
print("="*80)
print()
print(f"Use script 15 (validation) to analyze this solution in detail.")
print()
