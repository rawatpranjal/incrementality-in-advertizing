"""
Validation and interpretation of FPPE solution.

Compares optimal pacing multipliers to observed pacing and analyzes:
- Correlation and prediction accuracy
- Budget utilization patterns
- Allocation efficiency
- Segmentation by campaign characteristics

Input: market_*.npz, fppe_solution_*.npz
Output: validation_{date}.txt (detailed analysis)
"""

import numpy as np
import argparse

print("="*80)
print("FPPE SOLUTION VALIDATION")
print("="*80)
print()

# Parse arguments
parser = argparse.ArgumentParser(description='Validate FPPE solution')
parser.add_argument('--market', type=str, required=True, help='Market file (*.npz)')
parser.add_argument('--solution', type=str, required=True, help='Solution file (*.npz)')
args = parser.parse_args()

# Load data
print(f"Loading market from {args.market}...")
market = np.load(args.market, allow_pickle=True)

B = market['B']
V = market['V']
lambda_obs = market['lambda_obs']
date = str(market['date'])

N, M = V.shape

print(f"Market: {N} campaigns × {M} auctions")
print()

print(f"Loading solution from {args.solution}...")
solution = np.load(args.solution, allow_pickle=True)

lambda_opt = solution['lambda_optimal']
allocations = solution['allocations']
spend = solution['spend']
converged = bool(solution['converged'])
iterations = int(solution['iterations'])

print(f"Converged: {converged} after {iterations} iterations")
print()

# Overall metrics
print("="*80)
print("OVERALL COMPARISON")
print("="*80)
print()

corr = np.corrcoef(lambda_opt, lambda_obs)[0, 1]
rmse = np.sqrt(((lambda_opt - lambda_obs)**2).mean())
mae = np.abs(lambda_opt - lambda_obs).mean()

print(f"Correlation: {corr:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print()

print("Pacing Distribution:")
print(f"  Observed:  mean={lambda_obs.mean():.4f}, median={np.median(lambda_obs):.4f}")
print(f"  Optimal:   mean={lambda_opt.mean():.4f}, median={np.median(lambda_opt):.4f}")
print()

print("Budget-Constrained Campaigns (λ >= 0.95):")
print(f"  Observed: {(lambda_obs >= 0.95).sum()} ({(lambda_obs >= 0.95).mean()*100:.1f}%)")
print(f"  Optimal:  {(lambda_opt >= 0.95).sum()} ({(lambda_opt >= 0.95).mean()*100:.1f}%)")
print()

# Budget utilization
print("="*80)
print("BUDGET UTILIZATION ANALYSIS")
print("="*80)
print()

budget_util = spend / (B + 1e-9)
budget_util = np.clip(budget_util, 0, 2)

print("Overall:")
print(f"  Mean utilization: {budget_util.mean():.4f}")
print(f"  Median utilization: {np.median(budget_util):.4f}")
print(f"  % using >95%: {(budget_util >= 0.95).mean()*100:.1f}%")
print(f"  % using >50%: {(budget_util >= 0.50).mean()*100:.1f}%")
print()

# Why is budget utilization low?
print("Diagnostic: Why is budget utilization low?")
print(f"  Campaigns with B > 0: {(B > 0).sum()} ({(B > 0).mean()*100:.1f}%)")
print(f"  Campaigns with spend > 0: {(spend > 0).sum()} ({(spend > 0).mean()*100:.1f}%)")
print(f"  Campaigns with wins: {(allocations.sum(axis=1) > 0).sum()}")
print()

# Allocation analysis
print("="*80)
print("ALLOCATION ANALYSIS")
print("="*80)
print()

wins_per_campaign = allocations.sum(axis=1)
print(f"Auctions won per campaign:")
print(f"  Mean: {wins_per_campaign.mean():.2f}")
print(f"  Median: {np.median(wins_per_campaign):.2f}")
print(f"  Max: {wins_per_campaign.max():.0f}")
print(f"  Campaigns with 0 wins: {(wins_per_campaign == 0).sum()} ({(wins_per_campaign == 0).mean()*100:.1f}%)")
print()

# Market structure analysis
print("="*80)
print("MARKET STRUCTURE ANALYSIS")
print("="*80)
print()

# Sparsity
sparsity = (V == 0).mean()
print(f"Valuation matrix sparsity: {sparsity*100:.1f}%")

# Avg campaigns per auction
campaigns_per_auction = (V > 0).sum(axis=0)
print(f"Campaigns per auction (with V > 0):")
print(f"  Mean: {campaigns_per_auction.mean():.1f}")
print(f"  Median: {np.median(campaigns_per_auction):.1f}")
print(f"  Min: {campaigns_per_auction.min():.0f}")
print(f"  Max: {campaigns_per_auction.max():.0f}")
print()

# Avg auctions per campaign
auctions_per_campaign = (V > 0).sum(axis=1)
print(f"Auctions per campaign (with V > 0):")
print(f"  Mean: {auctions_per_campaign.mean():.1f}")
print(f"  Median: {np.median(auctions_per_campaign):.1f}")
print(f"  Min: {auctions_per_campaign.min():.0f}")
print(f"  Max: {auctions_per_campaign.max():.0f}")
print()

# Segmentation analysis
print("="*80)
print("SEGMENTATION ANALYSIS")
print("="*80)
print()

# Segment 1: Budget-constrained (observed λ >= 0.95)
bc_mask = lambda_obs >= 0.95
print(f"Budget-Constrained Campaigns (observed λ >= 0.95): {bc_mask.sum()}")
if bc_mask.sum() > 0:
    print(f"  Optimal λ: mean={lambda_opt[bc_mask].mean():.4f}, median={np.median(lambda_opt[bc_mask]):.4f}")
    print(f"  Budget util: mean={budget_util[bc_mask].mean():.4f}")
    print(f"  Wins: mean={wins_per_campaign[bc_mask].mean():.2f}")
    print()

# Segment 2: Unconstrained (observed λ < 0.95)
uc_mask = lambda_obs < 0.95
print(f"Unconstrained Campaigns (observed λ < 0.95): {uc_mask.sum()}")
if uc_mask.sum() > 0:
    print(f"  Optimal λ: mean={lambda_opt[uc_mask].mean():.4f}, median={np.median(lambda_opt[uc_mask]):.4f}")
    print(f"  Budget util: mean={budget_util[uc_mask].mean():.4f}")
    print(f"  Wins: mean={wins_per_campaign[uc_mask].mean():.2f}")
    print()

# Segment 3: High-budget campaigns
high_budget_mask = B > np.quantile(B, 0.75)
print(f"High-Budget Campaigns (top quartile): {high_budget_mask.sum()}")
if high_budget_mask.sum() > 0:
    print(f"  Budget range: ${B[high_budget_mask].min():.2f} - ${B[high_budget_mask].max():.2f}")
    print(f"  Observed λ: mean={lambda_obs[high_budget_mask].mean():.4f}")
    print(f"  Optimal λ: mean={lambda_opt[high_budget_mask].mean():.4f}")
    print(f"  Wins: mean={wins_per_campaign[high_budget_mask].mean():.2f}")
    print()

# Segment 4: Low-budget campaigns
low_budget_mask = B < np.quantile(B, 0.25)
print(f"Low-Budget Campaigns (bottom quartile): {low_budget_mask.sum()}")
if low_budget_mask.sum() > 0:
    print(f"  Budget range: ${B[low_budget_mask].min():.2f} - ${B[low_budget_mask].max():.2f}")
    print(f"  Observed λ: mean={lambda_obs[low_budget_mask].mean():.4f}")
    print(f"  Optimal λ: mean={lambda_opt[low_budget_mask].mean():.4f}")
    print(f"  Wins: mean={wins_per_campaign[low_budget_mask].mean():.2f}")
    print()

# Issues and recommendations
print("="*80)
print("DIAGNOSTICS & RECOMMENDATIONS")
print("="*80)
print()

issues = []

if corr < 0.3:
    issues.append("Low correlation between optimal and observed pacing")

if budget_util.mean() < 0.5:
    issues.append("Low budget utilization - campaigns not spending their budgets")

if (wins_per_campaign == 0).mean() > 0.5:
    issues.append("Many campaigns (>50%) win no auctions - market too sparse or budgets too low")

if sparsity > 0.95:
    issues.append("Very sparse valuation matrix (>95%) - limited competition")

if campaigns_per_auction.mean() < 5:
    issues.append("Low competition per auction (<5 campaigns on average)")

if len(issues) > 0:
    print("⚠️  Issues identified:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()

    print("Recommendations:")
    if sparsity > 0.95:
        print("  - Increase number of sampled auctions (--num_auctions in script 10)")
        print("  - Or: Focus on high-activity auctions with more bidders")

    if budget_util.mean() < 0.5:
        print("  - Check budget estimation method (may be overestimating)")
        print("  - Or: FPPE algorithm may need adjustment (more aggressive spending)")

    if (wins_per_campaign == 0).mean() > 0.5:
        print("  - Market structure issue: too many campaigns for too few auctions")
        print("  - Consider filtering to campaigns with higher activity")
else:
    print("✓ No major issues detected")

print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()

if corr > 0.5 and budget_util.mean() > 0.8:
    print("✓ FPPE solution is VALID")
    print("  - Good correlation with observed pacing")
    print("  - Reasonable budget utilization")
    print("  - Model captures market dynamics")
elif corr > 0.3:
    print("~ FPPE solution is PARTIAL")
    print("  - Moderate correlation with observed pacing")
    print("  - May capture some aspects of market behavior")
    print("  - Consider refinements to model or data")
else:
    print("✗ FPPE solution is QUESTIONABLE")
    print("  - Low correlation with observed pacing")
    print("  - Likely model specification or data issues")
    print("  - Investigate market structure and algorithm")

print()
