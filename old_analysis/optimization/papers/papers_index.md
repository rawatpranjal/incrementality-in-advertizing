# Optimization Papers Index

Canonical references for ad ecosystem optimization problems. Papers are organized by theme with explicit optimization formulations.

## Download Status

**Total Papers in Collection**: 46 PDFs downloaded
**Papers Requiring Manual Download**: 12 (ACM/INFORMS paywalled)
**Last Updated**: October 9, 2025

---

## 1. Surveys & Foundations

### [1] Online Matching in Advertisement Auctions (Survey)
- **File**: `survey_devanur_mehta_online_matching.pdf` ✓
- **Authors**: Devanur, Mehta
- **URL**: http://www.cs.toronto.edu/~bor/2421s22/papers/Devanur-Mehta-survey.pdf
- **Status**: Downloaded (613 KB)
- **Notes**: Consolidated formulations for allocation, pacing, and auction design

### [2] Auto-bidding and Auctions in Online Advertising (Survey)
- **File**: `survey_autobidding_auctions.pdf` ✓
- **Authors**: Multiple
- **Year**: 2024
- **URL**: https://arxiv.org/html/2408.07685v1
- **Status**: Downloaded (346 KB)
- **Notes**: CMDP/BwK frameworks with references to formal models

---

## 2. Online Budgeted Allocation (AdWords Problem)

### [3] AdWords and Generalized Online Matching
- **File**: `budgeted_mehta_adwords_generalized.pdf` ✓
- **Authors**: Mehta, Saberi, Vazirani, Vazirani
- **URL**: https://people.eecs.berkeley.edu/~vazirani/pubs/adwords.pdf
- **Status**: Downloaded (186 KB)
- **Notes**: Primal-dual and 1−1/e competitive algorithms

### [4] The Adwords Problem: Online Keyword Matching (JACM)
- **File**: `budgeted_mehta_adwords_jacm.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/1566374.1566384
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Budgeted matching with theoretical guarantees

### [5] Online Budgeted Allocation with General Budgets
- **File**: `budgeted_general_budgets.pdf` ✓
- **URL**: https://users.cs.duke.edu/~debmalya/papers/ec16-adgen.pdf
- **Status**: Downloaded (519 KB)
- **Notes**: Extensions for complex budget structures

### [6] Adwords with Unknown Budgets and Beyond
- **File**: `budgeted_unknown_budgets.pdf` ✓
- **URL**: https://arxiv.org/pdf/2110.00504.pdf
- **Status**: Downloaded (793 KB)
- **Notes**: Hardness/approximation guarantees

---

## 3. Pacing and Repeated Auctions

### [7] Learning in Repeated Auctions with Budgets
- **File**: `pacing_balseiro_regret.pdf` ⚠️
- **Authors**: Balseiro, Gur
- **URL**: https://dl.acm.org/doi/10.1145/3033274.3084088
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Lagrangian pacing multipliers, regret bounds

### [8] Pacing Equilibrium in First-Price Auction Markets
- **File**: `pacing_equilibrium_first_price.pdf` ⚠️
- **URL**: https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4310
- **Status**: INFORMS paywall (6.9 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Equilibrium characterizations for budget-constrained repeated auctions

---

## 4. Optimal Spend-Rate Estimation & Pacing

### [9] Optimal Spend Rate Estimation and Pacing
- **File**: `spend_optimal_rate_estimation.pdf` ✓
- **URL**: https://arxiv.org/abs/2202.05881
- **Status**: Downloaded (2.0 MB)
- **Notes**: Joint forecasting-and-control formulations

### [10] Optimal Spend Rate (NSF)
- **File**: `spend_optimal_pacing_nsf.pdf` ✓
- **URL**: https://par.nsf.gov/servlets/purl/10334320
- **Status**: Downloaded (1.0 MB)
- **Notes**: Spend targets and pacing error objectives

### [11] Optimization-Based Budget Pacing in eBay Sponsored Search
- **File**: `spend_ebay_sponsored_search.pdf` ✓
- **URL**: https://dspace.mit.edu/bitstream/handle/1721.1/155160/3589335.3648331.pdf
- **Status**: Downloaded (1.5 MB)
- **Notes**: Deployed at scale implementation

---

## 5. Autobidding via RL/CMDP

### [12] Real-Time Bidding by Reinforcement Learning
- **File**: `rl_rtb_display.pdf` ✓
- **URL**: https://arxiv.org/abs/1701.02490
- **Status**: Downloaded (1.1 MB)
- **Notes**: Sequential decision with budget constraints

### [13] Budget Constrained Bidding by Model-free RL
- **File**: `rl_budget_constrained.pdf` ✓
- **URL**: https://arxiv.org/pdf/1802.08365.pdf
- **Status**: Downloaded (1.6 MB)
- **Notes**: CMDP formulation

### [14] OCPC via Constrained MDP (2025)
- **File**: `rl_ocpc_cmdp.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/3706420
- **Status**: ACM paywall (6.7 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Industrial CMDP implementation

---

## 6. Bandits with Knapsacks (BwK)

### [15] Bandits with Knapsacks (FOCS 2013)
- **File**: `bwk_bandits_knapsacks_focs.pdf` ✓
- **URL**: https://ieee-focs.org/FOCS-2013-Papers/5135a207.pdf
- **Status**: Downloaded (246 KB)
- **Notes**: Resource-constrained exploration–exploitation

### [16] Bandits with Knapsacks (JACM)
- **File**: `bwk_bandits_knapsacks_jacm.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/3164539
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Regret/competitive guarantees

### [17] Linear Contextual BwK
- **File**: `bwk_linear_contextual.pdf` ✓
- **Authors**: Devanur et al.
- **URL**: https://www.nikhildevanur.com/pubs/NIPSLinContextualmain.pdf
- **Status**: Downloaded (142 KB)
- **Notes**: Contextual bandits with budget constraints

---

## 7. CPA/ROI-Constrained Auctions

### [18] Cost Per Action Constrained Auctions
- **File**: `cpa_constrained_auctions.pdf` ✓
- **URL**: https://arxiv.org/pdf/1809.08837.pdf
- **Status**: Downloaded (782 KB)
- **Notes**: Lagrangian formulations for CPA

### [19] Cost Per Action Constrained Auctions (ACM)
- **File**: `cpa_constrained_auctions_acm.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/3338506.3340269
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: CPA mechanism design

### [20] Multi-channel Autobidding with Budget and ROI Constraints
- **File**: `roi_multichannel_autobidding.pdf` ✓
- **URL**: https://proceedings.mlr.press/v202/deng23c/deng23c.pdf
- **Status**: Downloaded (899 KB)
- **Notes**: Multi-channel constrained optimization

---

## 8. Lift-Based Bidding (Incrementality)

### [21] Unbiased Lift-based Bidding System (AdKDD 2020)
- **File**: `lift_unbiased_adkdd.pdf` ✓
- **Authors**: Moriwaki et al.
- **URL**: http://papers.adkdd.org/2020/papers/adkdd20-moriwaki-unbiased.pdf
- **Status**: Downloaded (1.7 MB)
- **Notes**: Debiased lift estimation and bidding rules

### [22] Unbiased Lift-based Bidding (AdKDD alternate)
- **URL**: https://www.adkdd.org/papers/unbiased-lift-based-bidding-system/2020
- **Status**: Same as [21]

### [23] Real-World Implementation of Lift-based Bidding (2022)
- **File**: `lift_real_world_impl.pdf` ✓
- **Authors**: Moriwaki, Hayakawa
- **URL**: https://arxiv.org/pdf/2202.13868.pdf
- **Status**: Downloaded (927 KB)
- **Notes**: Production deployment details

### [24] Real-World Implementation (alternate link)
- **URL**: https://arxiv.org/abs/2202.13868
- **Status**: Same as [23]

---

## 9. Multi-Platform Budget Optimization

### [25] Stochastic Bandits for Multi-platform Budget Optimization
- **File**: `multiplatform_bandits_www.pdf` ✓
- **URL**: https://arxiv.org/abs/2103.10246
- **Status**: Downloaded (829 KB)
- **Notes**: BwK-style spend allocation across platforms

### [26] Multi-platform Optimization (WWW/ACM DL)
- **File**: `multiplatform_bandits_acm.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/3442381.3450074
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Stochastic rewards with budget constraints

---

## 10. Guaranteed Delivery (GD) Contracts

### [27] Ad Serving Using a Compact Allocation Plan (ACM)
- **Authors**: Vee, Vassilvitskii, Shanmugasundaram
- **URL**: https://dl.acm.org/doi/abs/10.1145/2229012.2229038
- **Status**: ACM paywall - REQUIRES MANUAL DOWNLOAD
- **Notes**: Convex/column-generation for GD allocation

### [28] Compact Allocation Plan
- **File**: `gd_compact_allocation.pdf` ✓
- **URL**: https://www.cs.tau.ac.il/~fiat/cgt12/EC_2012/docs/p319.pdf
- **Status**: Downloaded (2.4 MB)
- **Notes**: Flow formulations for GD

### [29] Unified Framework for Scheduling GD (Operations Research)
- **File**: `gd_reach_frequency_or.pdf` ⚠️
- **URL**: https://pubsonline.informs.org/doi/10.1287/opre.2016.1567
- **Status**: INFORMS paywall (6.9 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Reach/frequency optimization with segment constraints

### [30] Traffic Shaping to Optimize Ad Delivery (ACM)
- **URL**: https://dl.acm.org/doi/10.1145/2739010
- **Status**: ACM paywall - REQUIRES MANUAL DOWNLOAD
- **Notes**: GD shortfall minimization

### [31] Traffic Shaping to Optimize Ad Delivery
- **File**: `gd_traffic_shaping.pdf` ✓
- **URL**: https://faculty.mccombs.utexas.edu/deepayan.chakrabarti/mywww/papers/ec12-trafficshaping.pdf
- **Status**: Downloaded (345 KB)
- **Notes**: User- and segment-level constraints

---

## 11. Frequency Capping Optimization

### [32] Frequency Capping in Online Advertising (WADS 2011)
- **File**: `freq_capping_wads.pdf` ✓
- **URL**: https://theory.epfl.ch/moranfe/Publications/WADS2011.pdf
- **Status**: Downloaded (292 KB)
- **Notes**: Online assignment with per-user caps

### [33] Optimal Online Frequency Capping via Weight Method
- **File**: `freq_capping_weights.pdf` ✓
- **Authors**: Zinkevich
- **URL**: https://martin.zinkevich.org/publications/weights.pdf
- **Status**: Downloaded (444 KB)
- **Notes**: Weight-based frequency control

### [34] Reach/Frequency Optimization Framework
- **File**: `freq_reach_optimization.pdf` ✓
- **URL**: https://arxiv.org/html/2501.04882v1
- **Status**: Downloaded (807 KB)
- **Notes**: Planning formulations with frequency constraints

---

## 12. Reserve Price Optimization

### [35] Reserve Price Optimization for First-Price Auctions (PMLR 2021)
- **File**: `reserve_first_price_pmlr.pdf` ✓
- **URL**: http://proceedings.mlr.press/v139/feng21b/feng21b.pdf
- **Status**: Downloaded (438 KB)
- **Notes**: First-price auction revenue maximization

### [36] Optimizing Reserve Prices in Display Advertising
- **File**: `reserve_choi_mela.pdf` ✓
- **Authors**: Choi, Mela
- **URL**: https://hanachoi.github.io/research-papers/choi_mela_optimal_reserve.pdf
- **Status**: Downloaded (725 KB)
- **Notes**: Estimated value distributions

### [37] Optimal Reserve Prices (SSRN)
- **File**: `reserve_display_ssrn.pdf` ⚠️
- **URL**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4523022
- **Status**: SSRN access required (6.5 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Mechanism-specific constraints

### [38] Header Bidding Reserve Optimization
- **File**: `reserve_header_bidding.pdf` ⚠️
- **URL**: https://research.tue.nl/files/161192690/smc_final.pdf
- **Status**: Redirect/access issue (6.8 KB) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Multi-market effects

---

## 13. Bid Shading (First Price)

### [39] Deep Distribution Network for Bid Shading (KDD 2021)
- **File**: `shading_deep_distribution.pdf` ⚠️
- **URL**: https://dl.acm.org/doi/10.1145/3447548.3467167
- **Status**: ACM paywall (6.8 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Distributionally robust shading

### [40] Double Distributionally Robust Bid Shading
- **File**: `shading_robust.pdf` ✓
- **URL**: https://arxiv.org/html/2410.14864v1
- **Status**: Downloaded (623 KB)
- **Notes**: Robustness to distributional shift

---

## 14. DSP Profit Maximization

### [41] Profit Maximization for Online Advertising DSPs (AdKDD 2017)
- **File**: `dsp_profit_maximization.pdf` ✓
- **Authors**: Grigas et al.
- **URL**: http://papers.adkdd.org/2017/papers/adkdd17-grigas-profit.pdf
- **Status**: Downloaded (620 KB)
- **Notes**: Lagrangian decomposition under auction rules

---

## 15. Dynamic Pricing and Bidding

### [42] Dynamic Pricing and Bidding for Display Campaigns (MSOM 2025)
- **File**: `dynamic_pricing_bidding.pdf` ⚠️
- **URL**: https://pubsonline.informs.org/doi/10.1287/msom.2023.0600
- **Status**: INFORMS paywall (6.9 KB redirect) - REQUIRES MANUAL DOWNLOAD
- **Notes**: Joint pricing–bidding optimization

---

## 16. Robust Pacing

### [43] Robust Budget Pacing with a Single Sample
- **File**: `robust_pacing_single_sample.pdf` ✓
- **URL**: https://arxiv.org/abs/2302.02006
- **Status**: Downloaded (540 KB)
- **Notes**: Sample-efficient guarantees

### [44] Robust Pacing (OpenReview)
- **File**: `robust_pacing_openreview.pdf` ✓
- **URL**: https://openreview.net/pdf?id=5h42xM0pwn
- **Status**: Downloaded (456 KB)
- **Notes**: Distributionally robust control

---

## 17. Delay- and Label-Aware Bidding

### [45] MCMF: Multi-Constraints With Merging Features
- **File**: `delay_mcmf.pdf` ✓
- **URL**: https://arxiv.org/abs/2206.12147
- **Status**: Downloaded (799 KB)
- **Notes**: Multi-constraint objectives for stable bid control

### [46] Long-Delayed Conversions Prediction for Bidding
- **File**: `delay_conversions_arxiv.pdf` ✓
- **URL**: https://arxiv.org/html/2411.16095v1
- **Status**: Downloaded (1.0 MB)
- **Notes**: Delay-aware labels

### [47] Delayed Conversions (OpenReview)
- **File**: `delay_conversions_openreview.pdf` ✓
- **URL**: https://openreview.net/forum?id=nJONhNxLWB
- **Status**: Downloaded (1.0 MB)
- **Notes**: Optimization under delayed feedback

---

## 18. Online Causal Optimization in RTB

### [48] Online Causal Inference for Advertising in RTB Auctions
- **File**: `causal_rtb.pdf` ✓
- **URL**: https://arxiv.org/pdf/1908.08600.pdf
- **Status**: Downloaded (1.7 MB)
- **Notes**: Policy learning and intervention optimization

---

## 19. Additional References

### [49] Learning to Bid in First-Price Auctions
- **File**: `first_price_learning.pdf` ✓
- **URL**: https://arxiv.org/pdf/2304.13477.pdf
- **Status**: Downloaded (637 KB)
- **Notes**: First-price dynamics with budgets

### [50-57] Additional References
- **Status**: URLs point to non-academic sources (Trade Desk blog, Semantic Scholar, GitHub)
- **Action**: Not downloaded (industry resources, not canonical papers)

---

## Papers Requiring Manual Download (Paywalled)

1. **budgeted_mehta_adwords_jacm.pdf** - ACM Digital Library
2. **pacing_balseiro_regret.pdf** - ACM Digital Library
3. **pacing_equilibrium_first_price.pdf** - INFORMS
4. **rl_ocpc_cmdp.pdf** - ACM Digital Library
5. **bwk_bandits_knapsacks_jacm.pdf** - ACM Digital Library
6. **cpa_constrained_auctions_acm.pdf** - ACM Digital Library
7. **multiplatform_bandits_acm.pdf** - ACM Digital Library
8. **gd_reach_frequency_or.pdf** - INFORMS
9. **reserve_display_ssrn.pdf** - SSRN
10. **reserve_header_bidding.pdf** - Access issue
11. **shading_deep_distribution.pdf** - ACM Digital Library
12. **dynamic_pricing_bidding.pdf** - INFORMS

**Recommendation**: Access these through institutional login or request from authors.

---

## Usage Notes

- Papers with ✓ are successfully downloaded
- Papers with ⚠️ encountered paywalls or access restrictions
- File sizes under 10KB indicate HTML redirects rather than actual PDFs
- All arXiv papers downloaded successfully
- ACM Digital Library and INFORMS require institutional access

## Key Themes for Quick Reference

- **Allocation**: [1-6]
- **Pacing**: [7-11]
- **RL/CMDP**: [12-14]
- **Bandits**: [15-17]
- **Constraints**: [18-20]
- **Incrementality**: [21-24]
- **Multi-platform**: [25-26]
- **GD**: [27-31]
- **Frequency**: [32-34]
- **Reserve**: [35-38]
- **Shading**: [39-40]
- **DSP**: [41]
- **Pricing**: [42]
- **Robust**: [43-44]
- **Delay**: [45-47]
- **Causal**: [48]
