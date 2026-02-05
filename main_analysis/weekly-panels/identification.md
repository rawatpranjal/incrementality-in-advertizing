# Identification Strategy: Causal Effects of Advertising in a Staggered Adoption Setting

## 1. Objective and Research Question

The primary objective of this analysis is to estimate the causal effect of vendors commencing advertising on the platform on their key business outcomes. We operate in a staggered adoption setting, where vendors begin participating in ad auctions at different times over the 26-week sample period.

Our central research question is: **What is the average treatment effect on the treated (ATT) of winning a first auction on a vendor's weekly impressions, clicks, and Gross Merchandise Value (GMV)?**

## 2. The Empirical Challenge: Staggered Adoption

The staggered timing of advertising adoption invalidates the simple two-way fixed-effects (TWFE) OLS estimator. The TWFE model, typically specified as:

$$
\log(Y_{it} + 1) = \alpha_i + \gamma_t + \beta D_{it} + \epsilon_{it}
$$ 

where $Y_{it}$ is the outcome for vendor $i$ at time $t$, $\alpha_i$ is a vendor fixed effect, $\gamma_t$ is a week fixed effect, and $D_{it}$ is a treatment indicator, implicitly weights comparisons between early-treated and late-treated units. This can lead to biased estimates of $\beta$ if treatment effects are heterogeneous—a condition we expect to be true in our setting. This requires a more robust identification strategy.

## 3. Identification Strategy

To address the challenges of staggered adoption, we will employ the difference-in-differences (DiD) estimator developed by Callaway and Sant'Anna (2021). This method is robust to treatment effect heterogeneity and is the current best practice for this empirical setting.

### 3.1. Treatment and Control Definition

-   **Treatment Event**: We define treatment onset as the first week a vendor **wins any ad auction**. This is an unambiguous, observable event that marks a vendor's entry into the advertising marketplace.
-   **Treatment Group**: The treatment group consists of the 139,356 vendors (97.5% of the sample) who win at least one auction during the sample period. Vendors are grouped into cohorts $g$ based on the week of their first auction win.
-   **Control Group**: The primary control group ($C=0$) consists of the 3,564 vendors (2.5% of the sample) who **never win an auction** during the 26-week period. This group provides a clean, time-invariant baseline.

### 3.2. Key Identifying Assumption: Parallel Trends

Our identification hinges on the **parallel trends assumption**. Formally, for any treatment cohort $g$ and time $t$, this assumes:

$$ E[Y_{it}(0) - Y_{i,g-1}(0) | C_i = g] = E[Y_{it}(0) - Y_{i,g-1}(0) | C_i = 0] $$ 

where $Y_{it}(0)$ is the potential outcome for vendor $i$ at time $t$ had they not been treated, and $C_i=g$ indicates that vendor $i$ belongs to cohort $g$. In words, this assumes that in the absence of treatment, the average change in outcomes for vendors in cohort $g$ between period $g-1$ and $t$ would have been the same as the average change for the never-treated group.

## 4. Formal Estimands and Aggregation

The Callaway and Sant'Anna method proceeds by first estimating group-time average treatment effects and then aggregating them.

### 4.1. Group-Time Average Treatment Effects

The fundamental parameter of interest is the **group-time average treatment effect, `ATT(g,t)`**, defined as the average treatment effect for cohort $g$ at time $t$:

$$ ATT(g,t) = E[Y_{it}(1) - Y_{it}(0) | C_i=g] $$ 

Under the parallel trends assumption, this is identified and estimated via a 2x2 DiD comparison between cohort $g$ and the control group $C=0$ for time period $t$ relative to the pre-treatment period $g-1$.

### 4.2. Aggregate Estimands

The `ATT(g,t)` parameters are aggregated to produce two key summary estimands:

1.  **Overall Average Treatment Effect on the Treated (`ATT`)**: The ATT is a weighted average of the `ATT(g,t)`s across all treated groups and post-treatment time periods. It is formally defined as:

    $$ ATT = \sum_{g \in \mathcal{G}} w_g \sum_{t=g}^T w_{t|g} ATT(g,t) $$ 
    where $\mathcal{G}$ is the set of all treatment cohorts and $w_g$ and $w_{t|g}$ are weights based on the size of each cohort and the number of post-treatment periods.

2.  **Dynamic Treatment Effects (`θ(e)`)**: The event-study parameter $	heta(e)$ is the average treatment effect for units that have been treated for exactly $e$ periods. It is constructed by averaging all `ATT(g,t)` where the event time $e = t - g$:

    $$ \theta(e) = \frac{1}{N_e} \sum_{g \in \mathcal{G} : g+e \le T} \sum_{i:C_i=g} (Y_{i,g+e}(1) - Y_{i,g+e}(0)) $$ 
    where $N_e$ is the number of units observed at event time $e$. These parameters are crucial for assessing pre-trends ($e < 0$) and tracing the evolution of effects post-treatment ($e \ge 0$).

## 5. Validation and Robustness Checks

To ensure the credibility of our findings, we will perform two crucial validation exercises:

1.  **Parallel Pre-Trends Test**: We will estimate and plot `θ(e)` for all available pre-treatment periods. We will then perform a joint Wald test on the null hypothesis that all pre-treatment coefficients are jointly zero: $H_0: \theta(e)=0, \forall e < 0$. A p-value greater than 0.10 will provide strong support for our identification strategy.
2.  **Alternative Control Group**: As a key robustness check, we will replicate the entire analysis using **"not-yet-treated"** vendors as the control group instead of "never-treated" vendors. The results should be qualitatively and quantitatively similar.

## 6. Interpretation and Presentation of Results

The results will be presented in a manner that directly follows the causal logic of the advertising funnel. The primary visualizations will be event-study plots showing `θ(e)` for each outcome, and the primary tables will report the summary `ATT` estimates. The interpretation will begin by establishing the validity of the identification via the pre-trends test, followed by a narrative that connects the causal effects across the marketing funnel.
