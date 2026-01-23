# Canonical Optimization Problems in Ad Ecosystems

This directory contains a comprehensive LaTeX document extracting the workhorse optimization formulations from canonical papers in advertising optimization.

## Files

- **optimization_problems.tex** - Main LaTeX source file
- **optimization_problems.pdf** - Compiled PDF (15 pages, 286 KB)
- **causal_optimization.tex** - Supplementary document on causal optimization

## Document Structure

### 1. Common Notation (Section 1)
Unified notation across all optimization problems:
- Indices: $t$ (time), $i$ (advertiser), $j$ (user/impression), $k$ (arm/platform)
- Budgets: $B_i$, $B$
- Bids/Prices: $b_t$, $p_t$, $v_t$
- Allocation: $x_{ij}$
- Multipliers: $\mu$, $\lambda$
- Expectations: $\mathbb{E}$, $\mathbb{P}$

### 2. Optimization Problems by Theme

#### Section 2: Online Budgeted Allocation (AdWords)
- **Problem**: Maximize $\sum v_{ij} x_{ij}$ subject to budget $\sum c_{ij} x_{ij} \leq B_i$
- **Papers**: Mehta et al. (FOCS 2005), Devanur et al. (EC 2016)
- **Key Result**: 1-1/e competitive ratio via primal-dual

#### Section 3: Pacing and Repeated Auctions
- **Problem**: Maximize $\mathbb{E}[\sum v_t \omega_t]$ subject to budget $\mathbb{E}[\sum c_t \omega_t] \leq B$
- **Papers**: Balseiro-Gur (EC 2017), Conitzer et al. (MS 2022), Learning to Bid in First-Price Auctions (arXiv)
- **Key Tool**: Lagrangian pacing multiplier $\mu$, primal-dual with budget learning

#### Section 4: Optimal Spend-Rate Estimation
- **Problem**: Minimize $\mathbb{E}[(\sum c_t - B)^2 + \sum \phi_t(\rho_t)]$
- **Papers**: Karande et al. (arXiv 2022), Chen et al. (WWW 2024 - eBay AdaptivePacing)
- **Key Tool**: Stochastic control, adaptive $\mu$ learning, bid shading vs. throttling

#### Section 5: Autobidding via RL/CMDP
- **Problem**: Maximize $\mathbb{E}_\pi[\sum r_t]$ subject to $\mathbb{E}_\pi[\sum c_t] \leq B$
- **Papers**: Cai et al. (2017), Wu et al. (2018), Zhou et al. (CIKM 2025)
- **Key Tool**: Constrained MDP, Lagrangian Q-learning

#### Section 6: Bandits with Knapsacks (BwK)
- **Problem**: Maximize $\mathbb{E}[\sum r_{k_t}]$ subject to resource $\mathbb{E}[\sum c_{k_t}] \leq B$
- **Papers**: Badanidiyuru et al. (FOCS 2013, JACM 2018), Agrawal-Devanur (NIPS 2016)
- **Key Result**: $O(\sqrt{TK \log K})$ regret

#### Section 7: CPA/ROI-Constrained Auctions
- **Problem**: Maximize conversions subject to CPA $\mathbb{E}[\sum c_t]/\mathbb{E}[\sum y_t] \leq \alpha$
- **Papers**: Despotovic et al. (2018), Chen et al. (KDD 2019), Deng et al. (ICML 2023 - Multi-Channel ROI)
- **Key Tool**: Lagrangian formulation, dual multipliers, per-channel budget allocation

#### Section 8: Lift-Based Bidding (Incrementality)
- **Problem**: Maximize $\mathbb{E}[\sum \ell_t \omega_t]$ where $\ell_t$ is incremental effect
- **Papers**: Moriwaki et al. (AdKDD 2020, arXiv 2022)
- **Key Tool**: Debiased lift estimation via IPS/doubly robust

#### Section 9: Multi-Platform Budget Optimization
- **Problem**: Maximize $\sum x_k \mathbb{E}[r_k]$ subject to $\sum x_k \leq B$
- **Papers**: Hasu et al. (WWW 2021)
- **Key Tool**: BwK reduction, UCB policies

#### Section 10: Guaranteed Delivery (GD)
- **Problem**: Minimize deviation subject to demand $\sum_i x_{ij} = d_j$, supply $\sum_j x_{ij} \leq s_i$
- **Papers**: Vee et al. (EC 2010), Hojjat et al. (OR 2017), Chakrabarti-Vee (EC 2012 - Traffic Shaping)
- **Key Tool**: Flow LP, dual variables, compact allocation plan, traffic shaping for underdelivery

#### Section 11: Frequency Capping
- **Problem**: Maximize $\sum v_{ij} x_{ij}$ subject to frequency $\sum_j x_{ij} \leq f_i$
- **Papers**: Buchbinder et al. (WADS 2011), Zinkevich, Gao-Qiao (arXiv 2025 - k-Anonymity Reach)
- **Key Tool**: Multiplicative weights, online assignment, privacy-preserving reach optimization

#### Section 12: Reserve Price Optimization
- **Problem**: Maximize $R(r) = \mathbb{E}[\text{revenue}(r)]$ over reserve $r$
- **Papers**: Feng et al. (PMLR 2021), Choi-Mela (MS)
- **Key Result**: Monopoly pricing condition $r^* = (1-F(r^*))/f(r^*)$

#### Section 13: Bid Shading (First-Price)
- **Problem**: Maximize $\mathbb{E}[(v-b) \mathbb{1}(b \geq b_{-i})]$ with distributional robustness
- **Papers**: Chen et al. (KDD 2021), Wang et al. (arXiv 2024)
- **Key Tool**: Deep distribution networks, robust optimization

#### Section 14: DSP Profit Maximization
- **Problem**: Maximize $\sum (v_{it} - c_{it}) \omega_{it}$ subject to per-campaign budgets
- **Papers**: Grigas et al. (AdKDD 2017)
- **Key Tool**: Lagrangian decomposition

#### Section 15: Robust Pacing
- **Problem**: Maximize $\min_{\mathcal{D} \in \mathcal{U}} \mathbb{E}_\mathcal{D}[\sum r_t]$ subject to budget
- **Papers**: Balseiro et al. (arXiv 2023)
- **Key Tool**: Distributionally robust optimization, single-sample ambiguity sets

#### Section 16: Delay-Aware Bidding
- **Problem**: Maximize conversions with delayed feedback (observe $y_t$ at $t+\tau$)
- **Papers**: Zhao et al. (2022), Liu et al. (2024)
- **Key Tool**: Survival models, predictive imputation

#### Section 17: Online Causal Optimization
- **Problem**: Maximize $\mathbb{E}[\sum y_t(\pi(x_t))]$ learning from biased logs
- **Papers**: Dikkala et al. (arXiv 2019)
- **Key Tool**: IPS, doubly robust estimation, policy learning

### 3. Summary Table (Section 18)
Consolidated table mapping themes to canonical objectives and constraints.

## Compilation

```bash
cd /Users/pranjal/Code/marketplace-incrementality/optimization/latex
pdflatex optimization_problems.tex
pdflatex optimization_problems.tex  # Second pass for cross-refs
```

## Key Features

1. **Unified Notation**: All problems use consistent symbols
2. **Canonical Formulations**: Extracted directly from papers
3. **Clean Math**: Standard optimization notation (maximize, subject to)
4. **Minimal Commentary**: Objective, concise, statistician/economist audience
5. **Cross-References**: Labels and equations numbered for easy reference
6. **Bibliography**: Core papers cited

## Usage

- **Quick reference** for optimization problem structures
- **Teaching resource** for ad optimization courses
- **Implementation guide** for building bidding/pacing systems
- **Research baseline** for extending canonical formulations

## Statistics

- **15 pages** (including objective statement, notation, formulations, bibliography)
- **20 sections** covering major optimization themes
- **30+ optimization problems** with full formulations
- **2-3 line contextual introductions** for each section explaining problem setting, objective, constraints, and technical approach
- **Bibliography** with 42 core references covering all major themes

## Recent Additions

Added detailed formulations for:
- **First-Price Learning with Budgets**: Primal-dual algorithm for bid shading in first-price auctions under budget constraints
- **eBay AdaptivePacing**: Optimization-based budget pacing comparing throttling vs. bid shading approaches
- **Traffic Shaping for GD**: Joint optimization of content recommendation and ad serving to minimize underdelivery penalties
- **k-Anonymity Reach Optimization**: Privacy-preserving reach maximization under k-anonymity constraints (LinkedIn)
- **Multi-Channel ROI Constraints**: Budget allocation across multiple advertising platforms with global ROI targets
- **Linear Contextual BwK**: Confidence ellipsoid methods for contextual bandits with resource constraints

## Next Steps

To extend this document:
1. Add more papers from `/optimization/papers/` directory
2. Include algorithm pseudocode (using `algorithm` package)
3. Add complexity/regret bounds in theorem environments
4. Cross-link related formulations across sections
5. Expand bibliography with all 46 downloaded papers

## Notes

- Some papers are paywalled (ACM DL, INFORMS); formulations extracted from arXiv preprints or surveys
- Focus is on **optimization formulation**, not solution algorithms (can be added later)
- All LaTeX source is self-contained in single .tex file
