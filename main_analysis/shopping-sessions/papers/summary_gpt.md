# Literature Summaries (from Docling-extracted texts)

This document summarizes the key questions, data, methods, findings, and practical implications of each paper processed via Docling. Paths to the source PDFs and extracted text are included for traceability.

Note: The file `2507.15113v1.pdf` appears twice (duplicate copy). It is summarized once below.

---

## Click A, Buy B: Rethinking Conversion Attribution in E‑Commerce Recommendations

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/2507.15113v1.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/2507.15113v1.txt`

- Core question: How should conversion attribution handle sessions where users click product A but purchase product B (CABB), avoiding biases of last‑click attribution?
- Data: Large‑scale e‑commerce sessions on Meta’s platform (details in abstract and intro).
- Method:
  - Reframe conversion prediction as multitask learning with separate heads for CABA (Click A → Buy A) and CABB (Click A → Buy B).
  - Taxonomy‑aware collaborative filtering weights: map products to taxonomy leaves; learn category‑to‑category similarity from co‑engagement logs; amplify likely substitutes/complements, downweight coincidental cross‑category purchases.
- Findings (from abstract):
  - Offline: 13.9% reduction in normalized entropy vs last‑click baseline.
  - Online A/B: +0.25% lift in primary business metric.
- Implications: Attribution and training targets should distinguish on‑item vs cross‑item conversions; category‑aware weighting helps isolate informative CABB signals, improving both offline metrics and online conversions.

---

## Between Click and Purchase: Predicting Purchase Decisions Using Clickstream Data

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/Article_V2_updated.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/Article_V2_updated.txt`

- Core question: Can modeling sequences of viewing behaviors (actions taken on pages) improve purchase prediction versus page‑category sequences?
- Data: Clickstream from a European online travel agency (airline-ticket searches across 12 routes; registered users).
- Method: Represent browsing paths as sequences of viewing behaviors (e.g., filters used, exploration steps); predict next viewing behavior and use it to predict purchases.
- Findings: The viewing‑behavior approach improves purchase prediction accuracy relative to existing alternatives (abstract claims improved accuracy; no single figure reported in abstract).
- Implications: Using granular action sequences (beyond page categories) can better capture purchase intent heterogeneity; valuable for funnel‑stage targeting and dynamic interventions.

---

## Consumer Heterogeneity and Paid Search Effectiveness: A Large Scale Field Experiment (eBay)

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/BNT_ECMA_rev.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/BNT_ECMA_rev.txt`

- Core question: What is the causal effectiveness of paid search (SEM) for a large, well‑known brand?
- Data: Large‑scale randomized field experiments at eBay across U.S. geographies and keyword groups.
- Method: Experimental holdouts for brand and non‑brand keywords to measure causal lifts vs observational estimates.
- Findings (from abstract):
  - Non‑experimental estimates overstate returns; experimental returns are a fraction of those.
  - Brand‑keyword ads show no measurable short‑term benefits.
  - Non‑brand keywords: positive effects for new and infrequent users; frequent/loyal users account for most spend with minimal incremental impact, yielding negative average returns.
- Implications: Invest SEM budget in acquisition (new/infrequent users) and non‑brand queries; de‑emphasize brand terms for established brands; use experiments to calibrate attribution.

---

## Buying Reputation as a Signal of Quality: Evidence from an Online Marketplace (Taobao)

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/BuyingRep.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/BuyingRep.txt`

- Core question: Does paying buyers for informative feedback (reward‑for‑feedback, RFF) act as a credible signal of product quality?
- Data: Taobao RFF program, transactions across four categories, 13,018 sellers; item‑ and transaction‑level panel with ratings metadata.
- Method: Empirical tests of signaling predictions—high‑quality sellers select RFF (especially for items lacking feedback); measure sales and feedback outcomes.
- Findings (from abstract):
  - High‑quality products, especially those without established feedback, are more likely to adopt RFF.
  - RFF adoption increases sales by 36%.
- Implications: Feedback incentives can alleviate cold‑start and improve information quality; platforms may allow seller‑funded feedback rewards to surface high‑quality items.

---

## Using Clickstream Data to Improve Flash Sales Effectiveness

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/FlashSales_R1_web.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/FlashSales_R1_web.txt`

- Core question: How to forecast funnel behavior and optimize prices in short‑window, high‑uncertainty flash sales using clickstream data?
- Data: Large data from a leading flash‑sales firm; SKU‑level funnel events (visit → info view → purchase) across campaigns/products.
- Method: Hierarchical model reflecting funnel stages; decompose variation (lifecycle dynamics, campaign/product heterogeneity); compare to ML baselines; simulate responsive pricing.
- Findings: Best statistical performance vs ML alternatives; enables early identification of winners/losers and supports price updates; simulated revenue lifts and improved demand‑supply matching (abstract qualitative claims).
- Implications: Early‑cycle behavioral signals can guide dynamic pricing and inventory allocation under extreme demand volatility.

---

## Estimating the Causal Impact of Recommendation Systems from Observational Data (EC’15)

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/ec15_causal_impact_recommendations.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/ec15_causal_impact_recommendations.txt`

- Core question: How much activity do recommenders cause versus activity that would have occurred via other means?
- Data: 2.1M users’ Amazon browsing logs over 9 months; >4,000 focal products experiencing instantaneous shocks in direct traffic.
- Method: Instrumental variables identification using shocks: require shock in focal product’s direct traffic while recommended products’ direct traffic remains constant; estimate causal click‑through rate (local average treatment effect) via a Wald estimator.
- Findings: Although recommendation clicks account for a large share of traffic, at least 75% of these views would have occurred absent recommendations (i.e., naive counts overstate causal impact).
- Implications: Beware correlated demand; use natural experiments/IV designs to estimate incremental effects; naive attribution can greatly overstate recommender value.

---

## Modeling Cross‑Category Purchases in Sponsored Search Advertising

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/spillovers.pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/spillovers.txt`

- Core question: How do searches in one category spill over into purchases across categories during sponsored search sessions?
- Data: 6‑month panel from a nationwide retailer advertising on Google; hundreds of keywords; four categories (bath, bedding, kitchen, home decor).
- Method: Model latent category utility as intrinsic (own‑category) + extrinsic (joint purchases/cross‑category interdependence); estimate search‑to‑purchase mapping with hierarchical Bayesian methods.
- Findings:
  - Significant spillovers: users often purchase outside the initially searched category.
  - Spillovers are asymmetric across category pairs.
  - Positive cross‑category interdependence for retailer‑specific keywords; brand/generic keywords less likely to induce cross‑category purchases.
- Implications: Account for cross‑category effects in SEM budgeting and targeting; attribution at the session/basket level is crucial for accurate ROI.

---

## Attributing Conversions in a Multichannel Online Marketing Environment: An Empirical Model and a Field Experiment (JMR)

- Source PDF: `unified-session-position-analysis/shopping-sessions/papers/ssrn-2621304 (1).pdf`
- Extracted text: `unified-session-position-analysis/shopping-sessions/papers/docling_text/ssrn-2621304 (1).txt`

- Core question: How to attribute incremental conversion value across multiple online channels using user‑level path data?
- Data: Hospitality firm; individual touchpoint paths across channels (display, paid search, referral, email, affiliates), visits, and purchases.
- Method: Three‑level model: (1) channel consideration, (2) visits over time, (3) subsequent purchases; estimate carryover and spillover at visit and purchase stages; validate via a field study pausing paid search.
- Findings:
  - Significant carryover and spillover effects (e.g., email and display trigger search and referral visits; email leads to purchases via search).
  - Channel contributions differ substantially from last‑click or common heuristics.
  - Field study (paid search pause) validates model’s incremental impact estimates.
- Implications: Multi‑touch attribution should model dynamic cross‑channel effects; last‑click is biased; experimentation helps validate observational models.

---

## Cross‑Paper Takeaways for Session/Position Incrementality

- Last‑click bias and correlated demand are pervasive; multitouch or causal designs (IV, experiments) consistently yield lower, more realistic incremental effects.
- Cross‑item (CABB) and cross‑category spillovers are material and asymmetric; attribution and targeting must operate at session/basket level, not single item.
- Early funnel behavior (viewing actions) improves predictability and can power responsive interventions (pricing, ranking, messaging).
- Platform design levers (e.g., feedback incentives) can alleviate cold‑start and surface quality, impacting conversion pathways.

