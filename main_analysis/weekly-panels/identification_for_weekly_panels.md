# Identification Strategy for Vendor-Week Panel Analysis

## 1. Setting

We observe a panel of vendors on an e-commerce marketplace (Poshmark) over 26 weeks. Each vendor may begin advertising (entering the auction system) at a different point in time. The core question is whether advertising causally affects vendor outcomes: impressions, clicks, and gross merchandise value (GMV). This is a staggered adoption design: once a vendor begins advertising, they remain in the advertising system for subsequent periods. There are approximately 143,000 vendors and 846,000 vendor-week observations. Of these, 97.5% of vendors eventually adopt advertising at some point during the sample period, while 2.5% never win an auction and serve as a potential control group.

The analysis proceeds in two parts. Part (a) estimates an average click elasticity of revenue using a two-way fixed effects (TWFE) framework and extends it with vendor-specific random slopes and empirical Bayes shrinkage. Part (b) estimates the causal effect of advertising adoption using modern difference-in-differences estimators robust to treatment effect heterogeneity.

## 2. Part (a): Two-Way Fixed Effects and Vendor Heterogeneity

### 2.1 Model

The baseline TWFE model specifies:

$$\log(\text{revenue}_{vt} + 1) = \alpha_v + \gamma_t + \beta \log(\text{clicks}_{vt} + 1) + \epsilon_{vt}$$

where $\alpha_v$ is a vendor fixed effect, $\gamma_t$ is a week fixed effect, and $\beta$ is the average click elasticity. Standard errors are clustered at the vendor level. This model removes time-invariant vendor characteristics and common weekly shocks. The coefficient $\beta$ has a log-log elasticity interpretation: a 1% increase in clicks is associated with a $\beta$% increase in revenue.

### 2.2 Vendor-Specific Elasticities

We extend to a mixed-effects model with vendor-specific random slopes:

$$\log(\text{revenue}_{vt} + 1) = \beta_v \log(\text{clicks}_{vt} + 1) + \alpha_v + \gamma_t + \epsilon_{vt}$$

where $\beta_v$ is a vendor-specific click elasticity. We estimate these using an empirical Bayes approach that shrinks extreme estimates (from vendors with few observations) toward the global mean, reducing noise. The resulting distribution of $\beta_v$ allows us to compute vendor-level incremental Return on Ad Spend (iROAS).

### 2.3 Limitations of the TWFE Elasticity Model

The TWFE elasticity model in Section 2.1 estimates a correlational relationship within vendors over time. It does not isolate exogenous variation in clicks. The vendor fixed effects remove permanent differences across vendors, and the week fixed effects absorb common time shocks, but there may still be time-varying confounders at the vendor level (e.g., a vendor running a sale simultaneously increases organic traffic and advertising). The coefficient $\beta$ should be interpreted as a conditional association, not a causal elasticity. It is a useful descriptive baseline, but the causal analysis requires Part (b).

## 3. Part (b): Causal Effects via Staggered Difference-in-Differences

### 3.1 The Problem with TWFE for Causal Inference Under Staggered Adoption

When vendors adopt advertising at different times, a naive TWFE regression of the form

$$Y_{vt} = \alpha_v + \gamma_t + \beta D_{vt} + \epsilon_{vt}$$

where $D_{vt}$ is a binary indicator for whether vendor $v$ has begun advertising by week $t$, does not generally estimate a well-defined causal parameter when treatment effects are heterogeneous.

Goodman-Bacon (2021) proves that the TWFE estimator $\hat{\beta}_{DD}$ is a weighted average of all possible two-group, two-period (2x2) difference-in-differences estimators that can be formed from the data:

$$\hat{\beta}_{DD} = \sum_{k \neq j} s_{kj} \hat{\beta}_{kj}^{2x2} + \sum_{k} \sum_{\ell > k} s_{k\ell} \left[ \mu_{k\ell} \hat{\beta}_{k\ell}^{2x2,k} + (1 - \mu_{k\ell}) \hat{\beta}_{k\ell}^{2x2,\ell} \right]$$

The weights $s$ depend on group sizes and the variance of the treatment dummy within each pair, and $\mu_{k\ell}$ depends on the relative treatment timing. The weights are proportional to the treatment variance, which is highest for groups treated near the middle of the panel.

Three types of 2x2 comparisons arise: (i) a treated timing group versus the never-treated group, (ii) an early-treated group versus a later-treated group (using the later group as controls before their treatment), and (iii) a later-treated group versus an already-treated group (using the already-treated group as controls). The third type is problematic: if treatment effects change over time within the already-treated group, their changing outcomes contaminate the DD estimate. This is the source of potential negative weighting.

The TWFE estimand can be decomposed as:

$$\text{plim} \; \hat{\beta}_{DD} = \underbrace{VWATT}_{\text{estimand}} + \underbrace{VWCT}_{\text{common trends}} + \underbrace{\Delta ATT}_{\text{bias from time-varying effects}}$$

The $\Delta ATT$ term is zero only when treatment effects are constant over time. When effects evolve (as we expect in advertising, where effects may build or decay), TWFE yields a biased estimate that can even have the wrong sign.

de Chaisemartin and D'Haultfoeuille (2020) reach the same conclusion from a different angle. They show that the TWFE coefficient equals $\sum_{(g,t): D_{g,t}=1} w_{g,t} \Delta_{g,t}$ where $\Delta_{g,t}$ is the average treatment effect in cell $(g,t)$ and the weights $w_{g,t}$ can be negative. Negative weights arise for cells where a group has been treated for many periods and where many other groups are also treated. In staggered designs, this means later periods for early adopters are most likely to receive negative weights.

Sun and Abraham (2021) extend this analysis to dynamic event-study specifications with leads and lags. They show that even in dynamic TWFE regressions, the coefficient on a given lead or lag can be contaminated by treatment effects from other periods when effects are heterogeneous across cohorts. This invalidates the common practice of using pre-period coefficients to test for parallel trends, because non-zero pre-treatment estimates can arise purely from treatment effect heterogeneity rather than violations of parallel trends.

### 3.2 Application to Our Setting

In our vendor-week panel, we have strong reasons to expect treatment effect heterogeneity:

(a) Vendors adopt advertising at very different points in the 26-week window, creating 25 distinct treatment cohorts plus a never-treated group.

(b) Vendors differ substantially in size, category, and advertising intensity. Larger or more active vendors may experience different returns to advertising than smaller ones.

(c) Advertising effects likely evolve over time. Initial visibility gains may be large, while subsequent gains may plateau or even decline as budget is exhausted. Alternatively, effects may accumulate as the vendor builds a customer base.

(d) The very high adoption rate (97.5%) means only 2.5% of vendors form the never-treated control group. This small control pool makes the TWFE estimate particularly reliant on timing comparisons between treated groups, amplifying the problems described above.

For these reasons, we conduct a Goodman-Bacon decomposition as a diagnostic to quantify how much identifying variation comes from timing comparisons versus treated-vs-untreated comparisons, and to assess whether the TWFE estimate is a reliable summary statistic. We then proceed with modern estimators that are robust to heterogeneous effects.

### 3.3 The Goodman-Bacon Decomposition as Diagnostic

Following Goodman-Bacon (2021), we decompose the TWFE estimator into its constituent 2x2 DD estimates and their weights. This decomposition serves three purposes:

First, it reveals what fraction of the identifying variation comes from treated-vs-untreated comparisons versus timing comparisons. If timing comparisons dominate (likely in our case given the small never-treated group), the TWFE estimate is more susceptible to bias from time-varying treatment effects.

Second, it displays the individual 2x2 DD estimates against their weights, showing whether there is substantial heterogeneity in the component estimates and whether particular comparisons drive the overall result.

Third, it motivates the use of modern estimators by quantifying the extent of the problem. If the treated-vs-untreated comparisons and timing comparisons yield very different estimates, this is direct evidence that TWFE is not a reliable summary.

We implement the decomposition manually. For each pair of timing groups $k$ and $\ell$ (with $t_k^* < t_\ell^*$), we compute the three 2x2 DD estimates ($\hat{\beta}_{kU}^{2x2}$, $\hat{\beta}_{k\ell}^{2x2,k}$, $\hat{\beta}_{k\ell}^{2x2,\ell}$) and their weights ($s_{kU}$, $s_{k\ell} \mu_{k\ell}$, $s_{k\ell}(1 - \mu_{k\ell})$). We also compute the de Chaisemartin and D'Haultfoeuille weights $w_{g,t}$ and report the fraction of negative weights and the robustness diagnostic $\sigma_{fe}$.

### 3.4 Modern Difference-in-Differences Estimators

We employ three modern estimators, each addressing the TWFE problem from a different angle, following the "forward-engineering" approach advocated by Baker, Callaway, Cunningham, Goodman-Bacon, and Sant'Anna (2025). All three share the same fundamental insight: build estimators from clean 2x2 building blocks that never use already-treated units as controls.

#### 3.4.1 Callaway and Sant'Anna (2021)

The Callaway-Sant'Anna estimator is our primary approach. It identifies group-time average treatment effects, $ATT(g,t)$, for each cohort $g$ (defined by the week of first treatment) at each calendar time $t$:

$$ATT(g,t) = E[Y_{vt}(g) - Y_{vt}(0) \mid G_v = g]$$

where $Y_{vt}(g)$ is the potential outcome if vendor $v$ is first treated in week $g$, $Y_{vt}(0)$ is the untreated potential outcome, and $G_v = g$ indicates that vendor $v$ first receives treatment in week $g$.

**Identification.** Under the assumptions of irreversibility of treatment (Assumption 1), random sampling (Assumption 2), limited treatment anticipation with $\delta = 0$ (Assumption 3), and conditional parallel trends (Assumptions 4 or 5), the $ATT(g,t)$ is nonparametrically identified via:

$$ATT(g,t) = E[Y_t - Y_{g-1} \mid G_g = 1] - E[Y_t - Y_{g-1} \mid C = 1]$$

when using the never-treated group as controls (Assumption 4), or via analogous expressions using not-yet-treated groups (Assumption 5). The base period $g - 1$ is the most recent pre-treatment period for cohort $g$.

When pre-treatment covariates $X$ are available, identification can proceed via outcome regression, inverse probability weighting (IPW), or doubly-robust (DR) estimands (Theorem 1 in Callaway and Sant'Anna, 2021). The DR approach is preferred in practice because it is consistent if either the outcome regression or the propensity score model is correctly specified, but not necessarily both.

**Aggregation.** The potentially large number of $ATT(g,t)$ parameters are aggregated into interpretable summary measures:

(i) An overall ATT: $\theta_{OW} = \sum_g w_g \sum_{t \geq g} w_{t|g} ATT(g,t)$ with weights proportional to group size.

(ii) Event-study estimates: $\theta_{es}(e) = \sum_{g: g+e \leq T} P(G = g \mid G + e \leq T) \cdot ATT(g, g+e)$ for event time $e = t - g$. These trace out the treatment effect dynamics relative to treatment onset.

(iii) Cohort-specific effects: averaging over post-treatment periods for each cohort separately.

(iv) Calendar-time effects: averaging over treated cohorts at each calendar time.

**Pre-trends testing.** We compute event-study estimates for pre-treatment periods $e < 0$ and conduct a joint Wald test of $H_0: \theta(e) = 0 \; \forall \; e < 0$. Importantly, this pre-trends test is valid under heterogeneous treatment effects, unlike pre-trends tests based on TWFE event-study regressions (Sun and Abraham, 2021).

**Control group choice.** We run the estimation using both never-treated vendors (primary) and not-yet-treated vendors (robustness). When using never-treated vendors (Assumption 4), the comparison group is fixed across all cohorts and time periods. This does not restrict pre-treatment trends, which is an advantage. When using not-yet-treated vendors (Assumption 5), the comparison group changes for each cohort-time pair, which can increase precision (more comparison units) but does impose some restrictions on pre-treatment trends (see Marcus and Sant'Anna, 2020). Given our small never-treated group (2.5%), we present both and assess whether conclusions are robust.

#### 3.4.2 Sun and Abraham (2021)

Sun and Abraham propose an "interaction-weighted" estimator that avoids the contamination problem in TWFE event-study regressions. They define the cohort average treatment effect on the treated (CATT):

$$CATT_{e,\ell} = E[Y_{i,e+\ell} - Y_{i,e+\ell}(\infty) \mid E_i = e]$$

where $E_i$ is the cohort (timing of first treatment) and $Y_{i,t}(\infty)$ is the potential outcome under never being treated. They decompose the TWFE event-study coefficient $\hat{\mu}_\ell$ to show it is a linear combination of $CATT_{e,\ell'}$ across all cohorts $e$ and relative periods $\ell'$, with weights that are non-linear functions of the cohort distribution. This means $\hat{\mu}_\ell$ can include effects from other relative periods, and pre-treatment coefficients can appear significant even when parallel trends holds, purely due to treatment effect heterogeneity.

Their alternative estimator first runs cohort-specific 2x2 DDs using a "clean" control group (either never-treated or last-to-be-treated cohort), then averages across cohorts with shares as weights. This yields a weighted average of $CATT_{e,\ell}$ with interpretable, non-negative weights for each relative period $\ell$.

#### 3.4.3 Borusyak, Jaravel, and Spiess (2024)

The imputation estimator of Borusyak, Jaravel, and Spiess takes a different constructive approach. Under parallel trends (Assumption 1: $E[Y_{it}(0)] = \alpha_i + \beta_t$) and no anticipation (Assumption 2), the estimator proceeds in three steps:

(i) Fit the unit and period fixed effects $\hat{\alpha}_i$ and $\hat{\beta}_t$ using untreated observations only (never-treated units in all periods, and treated units in their pre-treatment periods).

(ii) Impute the untreated potential outcome for each treated observation: $\hat{Y}_{it}(0) = \hat{\alpha}_i + \hat{\beta}_t$.

(iii) Compute the treatment effect estimate for each treated observation: $\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)$.

(iv) Aggregate the $\hat{\tau}_{it}$ with pre-specified weights corresponding to the target estimand (e.g., ATT, or horizon-specific effects).

This estimator is the efficient linear unbiased estimator under spherical errors when treatment effect heterogeneity is unrestricted. It uses all pre-treatment periods for imputation, which gives it a precision advantage over estimators that use only the immediately preceding period. In our application with 26 weeks, this efficiency gain can be substantial.

The imputation estimator also provides a principled test of parallel trends: regress the residuals from step (i) on event-time indicators using only untreated observations. This test is not contaminated by treatment effect heterogeneity and avoids inference-after-testing problems under spherical errors.

### 3.5 Comparing Estimators

Baker et al. (2025) provide a unifying framework: all three modern estimators can be understood as constructing 2x2 "building blocks" that compare a set of newly-treated units with a set of not-yet-treated or never-treated units, then aggregating these building blocks with known, non-negative weights. The estimators differ in:

(a) Which building blocks they construct and which comparison group they use.

(b) How they aggregate building blocks across cohorts and time periods.

(c) Their statistical efficiency properties.

Callaway-Sant'Anna is the most flexible: it allows for covariates via doubly-robust estimation, it handles both never-treated and not-yet-treated comparison groups, and it provides a rich set of aggregation schemes. Sun-Abraham is regression-based and thus familiar to applied researchers, but does not handle covariates as naturally. Borusyak et al. is the most efficient under spherical errors but requires stronger parallel trends assumptions (the TWFE model must hold for all units, not just pairwise).

We present results from all three for robustness. If they agree, this reinforces confidence in the findings. If they disagree, the differences point to violations of specific assumptions (e.g., differences between never-treated and not-yet-treated comparisons suggest the never-treated group may not be a valid control).

## 4. Treatment Definition

We define treatment as the first week in which a vendor records a positive number of clicks, i.e., $G_v = \min\{t : \text{clicks}_{vt} > 0\}$. This marks the point at which the vendor's advertising generates measurable user engagement. If a vendor never records a click, they are assigned $G_v = \infty$ (never-treated).

As a robustness check, we also consider an alternative treatment definition based on impressions: $G_v = \min\{t : \text{impressions}_{vt} > 0\}$. This captures the first week the vendor's ads are shown, regardless of whether they generate clicks. The impressions-based definition will generally place vendors into earlier cohorts (since impressions precede clicks in the funnel). Comparing results across the two definitions helps assess whether the measured effects are driven by visibility (impressions) or engagement (clicks).

Treatment is absorbing by construction in this context: once a vendor has entered the advertising system and received clicks, they remain in the system. There may be weeks with zero clicks after the first positive week, but the vendor is still "treated" (present in the auction system). This is analogous to an intent-to-treat framework where treatment onset marks entry into the system, not continuous active engagement.

## 5. Identifying Assumptions

### 5.1 Parallel Trends (Unconditional)

The key identifying assumption for all estimators is parallel trends. For our primary specification using never-treated vendors as controls:

$$E[Y_{vt}(0) - Y_{v,g-1}(0) \mid G_v = g] = E[Y_{vt}(0) - Y_{v,g-1}(0) \mid G_v = \infty]$$

for all cohorts $g$ and post-treatment periods $t \geq g$. In words: absent advertising, the average change in outcomes for vendors who first advertise in week $g$ would have been the same as the average change for vendors who never advertise.

This is a substantive assumption. Vendors who adopt advertising may differ systematically from those who do not: they may be larger, growing faster, or selling in different categories. The assumption does not require that treated and never-treated vendors have the same level of outcomes. It requires only that their outcomes would have evolved similarly over time in the absence of treatment.

### 5.2 Conditional Parallel Trends

When unconditional parallel trends is implausible because the distribution of observable characteristics differs across cohorts, we can weaken it to conditional parallel trends:

$$E[Y_{vt}(0) - Y_{v,g-1}(0) \mid X_v, G_v = g] = E[Y_{vt}(0) - Y_{v,g-1}(0) \mid X_v, G_v = \infty]$$

where $X_v$ is a vector of pre-treatment covariates. This allows the outcome trends to depend on covariates, as long as after conditioning on $X_v$, the trends are parallel across treated and untreated groups.

Callaway and Sant'Anna (2021) is the only estimator among our three that natively accommodates conditional parallel trends, via doubly-robust estimands that model both the outcome regression and the generalized propensity score. The `differences` Python package implements this with a formula syntax: `formula = 'y ~ x1 + x2'`.

### 5.3 Covariate Selection and Pitfalls

Covariates must satisfy two conditions. First, they must be determined before treatment (pre-treatment covariates only). Conditioning on post-treatment variables can create endogeneity. In our setting, this means covariates should be measured in the period before the vendor first advertises (or at baseline). Candidates include:

(a) Pre-treatment vendor size (e.g., average pre-treatment GMV, number of products listed).

(b) Pre-treatment activity (e.g., number of weeks with positive organic sales before treatment).

(c) Product category or placement mix (e.g., proportion of auctions in Placement 1 vs. Placement 3).

Second, covariates must be plausibly related to the evolution of untreated potential outcomes. Including irrelevant covariates does not help and can reduce precision. Including covariates that are themselves affected by anticipation of treatment is harmful.

Baker et al. (2025) caution that covariates can create complications. When covariates differ substantially across treatment cohorts, the doubly-robust estimator may become noisy or rely heavily on extrapolation. The overlap assumption (Assumption 6 in Callaway and Sant'Anna) requires that the probability of being in any treatment cohort, conditional on covariates, is bounded away from zero and one. With many cohorts and many covariates, this can fail in practice.

For our initial analysis, we proceed without covariates (unconditional parallel trends). We then introduce covariates as a robustness check, using the `base_delta` parameter in the `differences` package to ensure that covariates are measured at a fixed pre-treatment time rather than varying with calendar time.

### 5.4 No Anticipation

We assume no anticipation with $\delta = 0$:

$$Y_{vt}(g) = Y_{vt}(0) \quad \forall t < g$$

This states that vendor outcomes before the first click are not affected by the future onset of advertising. This is plausible if vendors do not adjust their behavior in anticipation of receiving clicks. However, there may be mild anticipation: a vendor who has set up a campaign and is waiting for their first click might invest more in product listings. If anticipation is suspected, one can set $\delta = 1$ (allowing one period of anticipation), which shifts the base period back by one week.

### 5.5 Irreversibility of Treatment

Assumption 1 in Callaway and Sant'Anna (2021) requires that treatment is absorbing: once treated, a vendor remains treated. As discussed in Section 4, this holds by construction when treatment is defined as the first positive click. However, some vendors may have intermittent clicks (active in some weeks, inactive in others). The staggered adoption framework treats them as always treated after their first click, which is an intent-to-treat interpretation.

## 6. Pre-Trends Validation

We validate the parallel trends assumption using three approaches:

First, we estimate the full vector of event-study parameters $\theta(e)$ for pre-treatment periods $e < 0$. If parallel trends holds and there is no anticipation, all pre-treatment coefficients should be zero: $\theta(e) = 0$ for $e < 0$. We plot these coefficients with simultaneous confidence bands (not pointwise), following Callaway and Sant'Anna (2021), which correctly accounts for the multiple testing problem.

Second, we conduct a joint Wald test of $H_0: \theta(e) = 0 \; \forall \; e < 0$. A $p$-value above 0.10 provides support for the parallel trends assumption. We report this test for each outcome (impressions, clicks, GMV).

Third, we examine the Borusyak et al. (2024) pre-trend test, which is based on the residuals from the imputation step and is not contaminated by heterogeneous treatment effects. We assess whether results from the two testing procedures agree.

## 7. Outcomes and Interpretation

The primary outcomes are:

(a) Weekly impressions: the number of times a vendor's promoted products are shown.

(b) Weekly clicks: the number of clicks on a vendor's promoted products.

(c) Weekly total GMV: the total gross merchandise value (including both promoted and organic sales).

We trace effects through the advertising funnel. A positive effect on impressions establishes that advertising generates visibility. A positive effect on clicks establishes that this visibility translates into engagement. The key question is whether these effects propagate to GMV, or whether advertising merely redirects purchases that would have occurred organically.

The implied click-through rate (ATT for clicks / ATT for impressions) should be consistent with marketplace benchmarks (approximately 3%). Substantial deviations would warrant investigation.

A null effect on GMV despite positive effects on impressions and clicks has several possible interpretations: (i) advertising displaces organic discovery (the "cannibalization" hypothesis), (ii) there are delayed conversion effects beyond the weekly window, (iii) the treated vendors are too small for their GMV effects to be detected at conventional significance levels. The heterogeneity analysis by pre-treatment sales terciles can shed light on interpretation (iii).

## 8. Robustness Checks

### 8.1 Alternative Control Group

We estimate all models using both never-treated vendors and not-yet-treated vendors as the comparison group. Agreement between the two provides strong evidence for the validity of the design.

### 8.2 Alternative Treatment Definition

We compare the clicks-based treatment definition (primary) with the impressions-based definition (robustness). If the treatment effect on GMV differs meaningfully between the two, this indicates that the conversion from impressions to clicks is itself important for the GMV effect, rather than mere visibility.

### 8.3 Alternative Estimation Methods

We present the overall ATT from Callaway-Sant'Anna, Sun-Abraham, and Borusyak et al. side by side. Agreement across estimators, which make different assumptions and use different comparison strategies, is a strong robustness check.

### 8.4 Heterogeneity by Pre-Treatment Sales

We split vendors into terciles based on their average weekly GMV in the pre-treatment period and estimate the C&S ATT for each subgroup. This tests whether the null GMV result (if observed) is driven by noise in small vendors masking a real effect among large vendors.

### 8.5 Bacon Decomposition

We conduct the Goodman-Bacon decomposition of the naive TWFE estimate to quantify how much of the identifying variation comes from potentially problematic timing comparisons and to assess the magnitude of potential negative weighting.

## 9. References

Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., and Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A Practitioner's Guide. arXiv:2503.13323.

Borusyak, K., Jaravel, X., and Spiess, J. (2024). Revisiting Event-Study Designs: Robust and Efficient Estimation. Review of Economic Studies, 91(6), 3253-3294.

Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods. Journal of Econometrics, 225(2), 200-230.

de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way Fixed Effects Estimators with Heterogeneous Treatment Effects. American Economic Review, 110(9), 2964-2996.

Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. Journal of Econometrics, 225(2), 254-277.

Sun, L. and Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. Journal of Econometrics, 225(2), 175-199.
