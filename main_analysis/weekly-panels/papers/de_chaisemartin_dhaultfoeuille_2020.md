## Two-way fixed effects estimators with heterogeneous treatment effects ∗

Clément de Chaisemartin † Xavier D'Haultfœuille ‡

March 5, 2020

## Abstract

Linear regressions with period and group fixed effects are widely used to estimate treatment effects. We show that they estimate weighted sums of the average treatment effects (ATE) in each group and period, with weights that may be negative. Due to the negative weights, the linear regression coefficient may for instance be negative while all the ATEs are positive. We propose another estimator that solves this issue. In the two applications we revisit, it is significantly different from the linear regression estimator.

Keywords: linear regressions, fixed effects, heterogeneous treatment effects, difference-in-differences. JEL Codes: C21, C23

∗ We are very grateful to Olivier Deschêsnes, Guido Imbens, Peter Kuhn, Kyle Meng, Jesse Shapiro, Dick Startz, Doug Steigerwald, Clémence Tricaud, Gonzalo Vazquez-Bare, members of the UCSB econometrics research group, and seminar participants at Bergen, CIREQ Econometrics conference, CREST, Goteborg, Gothenburg, Groningen, ITAM, Pompeu Fabra, Stanford, SMU, Tinbergen Institute, UCL, UCLA, UC Davis, UCSB, USC, and Warwick for their helpful comments. Xavier D'Haultfœuille gratefully acknowledges financial support from the research grants Otelo (ANR-17-CE26-0015-041) and the Labex Ecodec: Investissements d'Avenir (ANR-11IDEX-0003/Labex Ecodec/ANR-11-LABX-0047).

† University of California at Santa Barbara, clementdechaisemartin@ucsb.edu

‡ CREST-ENSAE, xavier.dhaultfoeuille@ensae.fr

## 1 Introduction

A popular method to estimate the effect of a treatment on an outcome is to compare over time groups experiencing different evolutions of their exposure to treatment. In practice, this idea is implemented by estimating regressions that control for group and time fixed effects. Hereafter, we refer to those as two-way fixed effects (FE) regressions. We conducted a survey, and found that 20% of all empirical articles published by the American Economic Review (AER) between 2010 and 2012 have used a two-way FE regression to estimate the effect of a treatment on an outcome. When the treatment effect is constant across groups and over time, such regressions estimate that effect under the standard 'common trends' assumption. However, it is often implausible that the treatment effect is constant. For instance, the minimum wage's effect on employment may vary across US counties, and may change over time. This paper examines the properties of two-way FE regressions when the constant effect assumption is violated.

We start by assuming that all observations in the same ( g, t ) cell have the same treatment and that the treatment is binary, as is for instance the case when the treatment is a county-level law. We consider the regression of Y i,g,t , the outcome of unit i in group g at period t on group fixed effects, period fixed effects, and D g,t , the treatment in group g at period t . Let ̂ β fe denote the coefficient of D g,t , and let β fe denote its expectation. Under the common trends assumption, we show that β fe is equal to a weighted sum of the treatment effect in each treated ( g, t ) cell:

∆ g,t is the average treatment effect (ATE) in group g and period t and the weights W g,t s sum to one but may be negative. Negative weights arise because ̂ β fe is a weighted sum of several difference-in-differences (DID), which compare the evolution of the outcome between consecutive time periods across pairs of groups. However, the 'control group' in some of those comparisons may be treated at both periods. Then, its treatment effect at the second period gets differenced out by the DID, hence the negative weights.

<!-- formula-not-decoded -->

The negative weights are an issue when the ATEs are heterogeneous across groups or periods. Then, one could have that β fe is negative while all the ATEs are positive. For instance, 1 . 5 × 1 -0 . 5 × 4 , a weighted sum of 1 and 4 , is strictly negative. Using the data set of Gentzkow et al. (2011), we find that 40% of the weights attached to β fe are negative, so β fe is not robust to heterogeneous effects. 1

Researchers may want to know how serious that issue is in the application they consider. We show that conditional on all treatments, the absolute value of the expectation of β fe divided by

̂ 1 Gentzkow et al. (2011) do not estimate β fe , but β fd , the treatment coefficient in the first-difference regression defined below. 46% of the weights attached to β fd are strictly negative.

the standard deviation of the weights is equal to the minimal value of the standard deviation of the ATEs across the treated ( g, t ) cells under which the average treatment on the treated (ATT) may actually have the opposite sign than that coefficient. One can estimate that ratio to assess the robustness of the two-way FE coefficient. If that ratio is close to 0, that coefficient and the ATT can be of opposite signs even under a small and plausible amount of treatment effect heterogeneity. In that case, treatment effect heterogeneity would be a serious concern for the validity of that coefficient. On the contrary, if that ratio is very large, that coefficient and the ATT can only be of opposite signs under a very large and implausible amount of treatment effect heterogeneity.

Finally, we propose a new estimator, DID M , that is valid even if the treatment effect is heterogeneous over time or across groups. It estimates the average treatment effect across all the ( g, t ) cells whose treatment changes from t -1 to t . It relies on common trends assumptions on both potential outcomes. Those conditions are partly testable, and we propose a test that amounts to looking at pre-trends. This test differs from the standard event study pre-trends test (see Autor, 2003), which has been shown to be invalid when treatment effects are heterogeneous (see Abraham and Sun, 2018). We show that our estimator is asymptotically normal. We compute it in the data sets of Gentzkow et al. (2011) and Vella and Verbeek (1998), and in both cases we find that it is significantly different from ̂ β fe . 2 Our estimator can be used in applications where, for each pair of consecutive dates, there are groups whose treatment does not change. We estimate that this condition is satisfied for around 80% of the papers using two-way fixed effects regressions found in our survey of the AER.

Overall, our paper has implications for applied researchers estimating two-way fixed effects regressions. First, we recommend that they compute the weights attached to their regression and the ratio of | ̂ β fe | divided by the standard deviation of the weights. To do so, they can use the twowayfeweights Stata package that is available from the SSC repository. If many weights are negative, and if the ratio is not very large, we recommend that they compute our new estimator, using the fuzzydid and did\_multiplegt Stata packages, also available from the SSC repository (see de Chaisemartin et al., 2019, for explanations on how to use the former package).

We extend our results in several important directions. First, another commonly-used regression is the first-difference regression of Y g,t -Y g,t -1 , the change in the mean outcome in group g , on period fixed effects and on D g,t -D g,t -1 , the change in the treatment. We let β fd denote the expectation of the coefficient of D g,t -D g,t -1 . We show that under common trends, β fd also identifies a weighted sum of treatment effects, with potentially some negative weights. Second, in our Web Appendix we show that our results extend to fuzzy designs, where the treatment varies within ( g, t ) cells, and to two-way fixed effects regressions with a non-binary treatment and with covariates.

2 In both cases, our estimator is also significantly different from ̂ β fd .

Our paper is related to the DID literature. Our main result generalizes Theorem 1 in de Chaisemartin and D'Haultfœuille (2018). When the data has two groups and two periods, the WaldDID estimand considered therein is equal to β fe and β fd . Our results on β fe and β fd are thus extensions of that theorem to the case with multiple periods and groups. 3 Moreover, our DID M estimator is related to the Wald-TC estimator with many groups and periods proposed in de Chaisemartin and D'Haultfœuille (2018), and to the multi-period DID estimator proposed by Imai and Kim (2018). In Section 4, we explain the differences between those three estimators.

More recently, Borusyak and Jaravel (2017), Abraham and Sun (2018), Athey and Imbens (2018), Callaway and Sant'Anna (2018), and Goodman-Bacon (2018) study the special case of staggered adoption designs, where the treatment of a group is weakly increasing over time. Those papers derive some important results specific to that design that we do not consider here. Still, some of the results in those papers are related to ours, and we describe precisely those connections later in the paper. The most important dimension on which our paper differs from those is that our results apply to any two-way fixed effects regressions, not only to those with staggered adoption. In our survey of the AER papers estimating two-way fixed effects regressions, less than 10% have a staggered adoption design. This suggests that while staggered adoptions are an important research design, they may account for a relatively small minority of the applications where two-way fixed effects regressions have been used.

The paper is organized as follows. Section 2 introduces the set-up. Section 3 presents our decomposition results. Section 4 introduces our alternative estimator. Section 5 briefly describes some of the extensions covered in our Web Appendix. Section 6 presents our survey of the articles published in the AER, and our two empirical applications.

## 2 Set up

One considers observations that can be divided into G groups and T periods. For every ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } , let N g,t denote the number of observations in group g at period t , and let N = ∑ g,t N g,t be the total number of observations. The data may be an individual-level panel or repeated cross-section data set where groups are, say, individuals' county of birth. The data could also be a cross-section where cohort of birth plays the role of time. For instance, Duflo (2001) compares the schooling of different cohorts in Indonesia, some of which were exposed to a school construction program. It is also possible that for all ( g, t ) , N g,t = 1 , e.g. a group is one individual or firm. All of the above are special cases of the data structure we consider.

One is interested in measuring the effect of a treatment on some outcome. Throughout the paper

3 In fact, a preliminary version of our main result appeared in a working paper version of de Chaisemartin and D'Haultfœuille (2018) (see Theorems S1 and S2 in de Chaisemartin and D'Haultfoeuille, 2015).

we assume that treatment is binary, but our results apply to any ordered treatment, as we show in Section ?? of the Web Appendix. Then, for every ( i, g, t ) ∈ { 1 , ..., N g,t }×{ 1 , ..., G }×{ 1 , ..., T } , let D i,g,t and ( Y i,g,t (0) , Y i,g,t (1)) respectively denote the treatment status and the potential outcomes without and with treatment of observation i in group g at period t .

The outcome of observation i in group g and period t is Y i,g,t = Y i,g,t ( D i,g,t ) . For all ( g, t ) , let

<!-- formula-not-decoded -->

D g,t denotes the average treatment in group g at period t , while Y g,t (0) , Y g,t (1) , and Y g,t respectively denote the average potential outcomes without and with treatment and the average observed outcome in group g at period t .

Throughout the paper, we maintain the following assumptions.

Assumption 1 (Balanced panel of groups) For all ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } , N g,t &gt; 0 .

Assumption 1 requires that no group appears or disappears over time. This assumption is often satisfied. Without it, our results still hold but the notation becomes more complicated as the denominators of some of the fractions below may then be equal to zero.

Assumption 2 (Sharp design) For all ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } and i ∈ { 1 , ..., N g,t } , D i,g,t = D g,t .

Assumption 2 requires that units' treatments do not vary within each ( g, t ) cell, a situation we refer to as a sharp design. This is for instance satisfied when the treatment is a group-level variable, for instance a county- or a state-law. This is also mechanically satisfied when N g,t = 1 . In our survey in Section 6.1, we find that almost 80% of the papers using two-way fixed effects regressions and published in the AER between 2010 and 2012 consider sharp designs. We focus on sharp designs because of their prevalence, but in Section ?? of the Web Appendix, we show that all the results in Sections 3-4 below can be extended to fuzzy designs.

Assumption 3 (Independent groups) The vectors ( Y g,t (0) , Y g,t (1) , D g,t ) 1 ≤ t ≤ T are mutually independent.

We consider D g,t , Y g,t (0) , Y g,t (1) as random variables. For instance, aggregate random shocks may affect the average potential outcomes of group g at period t . The treatment status of group g at period t may also be random. The expectations below are taken with respect to the distribution of those random variables. Assumption 3 allows for the possibility that the treatments and potential outcomes of a group may be correlated over time, but it requires that the potential outcomes and treatments of different groups be independent.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 4 requires that the shocks affecting a group's Y g,t (0) be mean independent of that group's treatment sequence. This rules out the possibility that a group gets treated because it experiences negative shocks, the so-called Ashenfelter's dip (see Ashenfelter, 1978). Assumption 4 is related to the strong exogeneity condition in panel data models, which, as is well-known, is necessary to obtain the consistency of the fixed effects estimator (see, e.g., Wooldridge, 2002).

We now define the FE regression described in the introduction. 4

Regression 1 (Fixed-effects regression)

Let ̂ β fe denote the coefficient of D g,t in an OLS regression of Y i,g,t on group fixed effects, period fixed effects, and D g,t . Let β fe = E [ ̂ β fe ] . 5

For all g and t , let N g,. = ∑ T t =1 N g,t and N .,t = ∑ G g =1 N g,t respectively denote the total number of observations in group g and in period t . For any variable X g,t defined in each ( g, t ) cell, let X g,. = ∑ T t =1 ( N g,t /N g,. ) X g,t denote the average value of X g,t in group g , let X .,t = ∑ G g =1 ( N g,t /N .,t ) X g,t denote the average value of X g,t in period t , and let X .,. = ∑ g,t ( N g,t /N ) X g,t denote the average value of X g,t . For instance, D 3 ,. and D ., 2 respectively denote the average treatment in group 3 across time and in period 2 across groups, whereas Y .,. denotes the average value of the outcome across groups and time. Finally, for any variable X g,t , we let X denote the vector ( X g,t ) ( g,t ) ∈{ 1 ,...,G }×{ 1 ,...,T } collecting the values of that variable in each ( g, t ) cell. For instance, D is the vector ( D g,t ) ( g,t ) ∈{ 1 ,...,G }×{ 1 ,...,T } collecting the treatments of all the ( g, t ) cells.

## 3 Two-way fixed effects regressions

## 3.1 A decomposition result

We study the FE regression under the following common trends assumption.

Assumption 5 (Common trends) For t ≥ 2 , E ( Y g,t (0) -Y g,t -1 (0)) does not vary across g .

Assumption 5 requires that the expectation of the outcome without treatment follow the same evolution over time in every group. When t represents birth cohorts, Assumption 5 requires that the outcome difference between consecutive cohorts be the same across groups.

4 Throughout the paper, we assume that D g,t in Regression 1 and D g,t -D g,t -1 in Regression 2 below are not collinear with the other independent variables in those regressions, so β fe and β fd are well-defined.

̂ ̂ 5 As the independent variables in Regression 1 are constant within each ( g, t ) cell, Regression 1 is equivalent to a ( g, t ) -level regression of Y g,t on group and period fixed effects and D g,t , weighted by N g,t .

Let N 1 = ∑ i,g,t D i,g,t denote the number of treated units, let denote the average treatment effect across all treated units, and let δ TR = E [ ∆ TR ] denote the expectation of that parameter, hereafter referred to as the ATT. For any ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

denote the ATE in cell ( g, t ) . δ TR is equal to the expectation of a weighted average of the treated cells' ∆ g,t s:

Under the common trends assumption, we show that β fe is also equal to the expectation of a weighted sum of the ∆ g,t s, with potentially some negative weights.

Let ε g,t denote the residual of observations in cell ( g, t ) in the regression of D g,t on group and period fixed effects: 6

<!-- formula-not-decoded -->

One can show that if the regressors in Regression 1 are not collinear, the average value of ε g,t across all treated ( g, t ) cells differs from 0: ∑ ( g,t ): D g,t =1 ( N g,t /N 1 ) ε g,t = 0 . Then we let w g,t denote ε g,t divided by that average:

Theorem 1 Suppose that Assumptions 1-5 hold. Then, 7

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6 ε g,t arises from a unit-level regression, where the dependent and independent variables only vary at the ( g, t ) level. Therefore, all the units in the same ( g, t ) cell have the same value of ε g,t .

7 In the proof, we show the following, stronger result:

<!-- formula-not-decoded -->

glyph[negationslash]

thus implying that glyph[negationslash]

This result implies that in general, β fe = δ TR , so ̂ β fe is a biased estimator of the ATT. To illustrate this, we consider a simple example of a staggered adoption design with two groups and three periods, and where the treatments are non-stochastic: group 1 is untreated at periods 1 and 2 and treated at period 3, while group 2 is untreated at period 1 and treated both at periods 2 and 3. 8 We also assume that N g,t /N g,t -1 does not vary across g : all groups experience the same growth of their number of observations from t -1 to t , a requirement that is for instance satisfied when the data is a balanced panel. Then, one can show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The residual is negative in group 2 and period 3, because the regression predicts a treatment probability larger than one in that cell, a classic extrapolation problem with linear regressions. Then, under the common trends assumption, it follows from Theorem 1 and the fact that the treatments are non-stochastic that

<!-- formula-not-decoded -->

β fe is equal to a weighted sum of the ATEs in group 1 at period 3, group 2 at period 2, and group 2 at period 3, the three treated ( g, t ) cells. However, the weight assigned to each ATE differs from 1 / 3 , the proportion that each cell accounts for in the population of treated observations. Therefore, β fe is not equal to δ TR . Perhaps more worryingly, not all the weights are positive: the weight assigned to the ATE in group 2 period 3 is strictly negative. Consequently, β fe may be a very misleading measure of the treatment effect. Assume for instance that E [∆ 1 , 3 ] = E [∆ 2 , 2 ] = 1 and E [∆ 2 , 3 ] = 4 . At the period when they start receiving the treatment, both groups experience a modest positive ATE. But this effect builds over time and in period 3, one period after it has started receiving the treatment, group 2 now experiences a large ATE. Then,

<!-- formula-not-decoded -->

β fe is strictly negative, while E [∆ 1 , 3 ] , E [∆ 2 , 2 ] , and E [∆ 2 , 3 ] are all positive. More generally, the negative weights are an issue if the E [∆ g,t ] s are heterogeneous, across groups or over time. 9 If E [∆ 1 , 3 ] = E [∆ 2 , 2 ] = E [∆ 2 , 3 ] = 1 , then β fe = 1 = δ TR .

8 A similar example appears in Borusyak and Jaravel (2017).

9 On the other hand, β fe does not rule out heterogeneous treatment effects within ( g, t ) cells, as it is identified by variations across ( g, t ) cells, and does not leverage any within-cell variation.

Here is some intuition as to why one weight is negative in this example. It follows from Equation (7) in the proof of Theorem 1 (see also Theorem 1 in Goodman-Bacon, 2018) that in this simple example, β fe = ( DID 1 + DID 2 ) / 2 , with

<!-- formula-not-decoded -->

The first DID compares the evolution of the mean outcome from period 1 to 2 in group 2 and in group 1. The second one compares the evolution of the mean outcome from period 2 to 3 in group 1 and in group 2. The control group in the second DID, group 2, is treated both in the pre and in the post period. Therefore, under the common trends assumption, it follows from Lemma 1 in Appendix A (a similar result appears in Lemma 1 of de Chaisemartin (2011) and in Equation (13) of Goodman-Bacon (2018)) that DID 1 = E [∆ 2 , 2 ] , but

<!-- formula-not-decoded -->

DID 2 is equal to the ATE in group 1 period 3, minus the change in group 2's ATE between periods 2 and 3. Intuitively, the mean outcome of groups 1 and 2 may follow different trends from period 2 to 3 either because group 1 becomes treated, or because group 2's ATE changes. The intuition that negative weights arise because ̂ β fe uses treated observations as controls also appears in Borusyak and Jaravel (2017).

We now generalize the previous illustration by characterizing the ( g, t ) cells whose ATEs are weighted negatively by β fe .

Proposition 1 Suppose that Assumption 1 holds and for all t ≥ 2 N g,t /N g,t -1 does not vary across g . Then, for all ( g, t, t ′ ) such that D g,t = D g,t ′ = 1 , D .,t &gt; D .,t ′ implies w g,t &lt; w g,t ′ . Similarly, for all ( g, g ′ , t ) such that D g,t = D g ′ ,t = 1 , D g,. &gt; D g ′ ,. implies w g,t &lt; w g ′ ,t .

Proposition 1 shows that β fe is more likely to assign a negative weight to periods where a large fraction of groups are treated, and to groups treated for many periods. Then, negative weights are a concern when treatment effects differ between periods with many versus few treated groups, or between groups treated for many versus few periods.

Proposition 1 has interesting implications in staggered adoption designs, a special case of sharp designs defined as follows.

Assumption 6 (Staggered adoption designs) For all g , D g,t ≥ D g,t -1 for all t ≥ 2 .

Assumption 6 is satisfied in applications where groups adopt a treatment at heterogeneous dates (see e.g. Athey and Stern, 2002). In that design, Borusyak and Jaravel (2017) show that β fe is more likely to assign a negative weight to treatment effects at the last periods of the panel.

This result is a special case of Proposition 1: in staggered adoption designs, D .,t is increasing in t , so Proposition 1 implies that w g,t is decreasing in t . 10 Proposition 1 also implies that in that design, groups that adopt the treatment earlier are more likely to receive some negative weights.

Finally, in staggered adoption designs, Athey and Imbens (2018) derive a decomposition of β fe that resembles to, but differs from, that in Theorem 1. They derive their decomposition under the assumption that the dates at which each group starts receiving the treatment are randomly assigned, while we derive ours under a common trends assumption.

## 3.2 Robustness to heterogeneous treatment effects

Theorem 1 shows that in sharp designs with many groups and periods, ̂ β fe may be a misleading measure of the treatment effect under the standard common trends assumption, if the treatment effect is heterogeneous across groups and time periods. In the corollary below, we propose two robustness measures that can be used to assess how serious that concern is.

Those robustness measures are defined conditional on D , the vector stacking together the treatments of all the ( g, t ) cells. Specifically, for all ( g, t ) ∈ { 1 , ..., G }×{ 1 , ..., T } , let ˜ ∆ g,t = E (∆ g,t | D ) denote the ATE in cell ( g, t ) conditional on D , 11 let ˜ ∆ TR = E ( ∆ TR ∣ ∣ D ) denote the ATT conditional on D , and let ˜ β fe = E ( ̂ β fe ∣ ∣ ∣ D ) . The first measure we consider is the minimal value of the standard deviation of the ˜ ∆ g,t s under which one could have that ˜ β fe is of a different sign than ˜ ∆ TR . Therefore, this summary measure applies to ˜ β fe and ˜ ∆ TR , rather than β fe and δ TR , the unconditional expectations of ̂ β fe and ∆ TR on which we have focused so far. However, one can show that when G , the number of groups, goes to infinity, ˜ β fe -β fe and ˜ ∆ TR -δ TR both converge to 0. So if the number of groups is large, ˜ β fe and ˜ ∆ TR should not differ much from β fe and δ TR , and our robustness measure 'almost' applies to β fe and δ TR .

<!-- formula-not-decoded -->

Let

10 Borusyak and Jaravel (2017) assume that the treatment effect of cell ( g, t ) only depends on the number of periods since group g has started receiving the treatment, whereas Proposition 1 does not rely on that assumption.

11 ˜ ∆ g,t may differ from E (∆ g,t ) . To see this, let us consider a simple example where T = 2 . Then, under Assumption 3, one has ˜ ∆ g,t = E (∆ g,t | D g, 1 , D g, 2 ) . One may for instance have E (∆ g, 1 | D g, 1 = 0 , D g, 2 = 0) &lt; E (∆ g, 1 | D g, 1 = 1 , D g, 2 = 1) , if a group is more likely to be treated if her treatment effect is initially high.

σ ( ˜ ∆ ) is the standard deviation of the conditional ATEs, and σ ( w ) is the standard deviation of the w -weights, 12 across the treated ( g, t ) cells. Let n = # { ( g, t ) : D g,t = 1 } denote the number of treated cells. For every i ∈ { 1 , ..., n } , let w ( i ) denote the i th largest of the weights of the treated cells: w (1) ≥ w (2) ≥ ... ≥ w ( n ) , and let N ( i ) and ˜ ∆ ( i ) be the number of observations and the conditional ATE of the corresponding cell. Then, for any k ∈ { 1 , ..., n } , let P k = ∑ i ≥ k N ( i ) /N 1 , S k = ∑ i ≥ k ( N ( i ) /N 1 ) w ( i ) and T k = ∑ i ≥ k ( N ( i ) /N 1 ) w 2 ( i ) .

Corollary 1 Suppose that Assumptions 1-5 hold.

1. If σ ( w ) &gt; 0 , the minimal value of σ ( ˜ ∆ ) compatible with ˜ β fe and ˜ ∆ TR = 0 is

glyph[negationslash]

<!-- formula-not-decoded -->

2. If ˜ β fe = 0 and at least one of the w g,t weights is strictly negative, the minimal value of σ ( ˜ ∆ ) compatible with ˜ β fe and with ∆ g,t of a different sign than ˜ β fe for all ( g, t ) is

where s = min { i ∈ { 1 , ..., n } : w ( i ) &lt; -S ( i ) / (1 -P ( i ) ) } .

<!-- formula-not-decoded -->

σ fe and σ fe can be estimated simply by replacing ˜ β fe by ̂ β fe . An estimator of σ fe can be used to assess the robustness of ̂ β fe to treatment effect heterogeneity across groups and periods. If σ fe is close to 0, ˜ β fe and ˜ ∆ TR can be of opposite signs even under a small and plausible amount of treatment effect heterogeneity. In that case, treatment effect heterogeneity would be a serious concern for the validity of ̂ β fe . On the contrary, if σ fe is very large, ˜ β fe and ˜ ∆ TR can only be of opposite signs under a very large and implausible amount of treatment effect heterogeneity. Then, treatment effect heterogeneity is less of a concern.

Similarly, if σ fe is close to 0, one may have, say, ˜ β fe &gt; 0 , while ˜ ∆ g,t ≤ 0 for all ( g, t ) , even if the dispersion of the ˜ ∆ g,t s across ( g, t ) cells is relatively small. Notice that σ fe is only defined if at least one of the weights is strictly negative: if all the weights are positive, then one cannot have that β fe is of a different sign than all the ∆ g,t s.

Assumption 7 ( w uncorrelated with ˜ ∆ ) E [ ∑ ( g,t ): D g,t =1 N g,t N 1 ( w g,t -1)( ˜ ∆ g,t -˜ ∆ TR ) ] = 0 . Corollary 2 If Assumptions 1-5 and 7 hold, then β fe = δ TR .

˜ ˜ When some of the weights w g,t are negative, ̂ β fe may still be robust to heterogeneous treatment effects across groups and periods, provided the assumption below is satisfied.

12 One can show that ∑ ( g,t ): D g,t =1 ( N g,t /N 1 ) w g,t = 1 .

Assumption 7 requires that the weights attached to the fixed effects estimator be uncorrelated with the conditional ATEs in the treated ( g, t ) cells. This is often implausible. For instance, groups treated the most are also those with the lowest value of w g,t , as shown in Proposition 1. But those groups could also be those with the largest treatment effect. This would then induce a negative correlation between w and ˜ ∆ . The plausibility of Assumption 7 can be assessed, by looking at whether w is correlated with a predictor of the treatment effect in each ( g, t ) cell. In the two applications we revisit in Section 6, this test is rejected.

## 3.3 Extension to the first-difference regression

Instead of Regression 1, many articles have estimated the first-difference regression defined below:

Regression 2 (First-difference regression)

Let ̂ β fd denote the coefficient of D g,t -D g,t -1 in an OLS regression of Y g,t -Y g,t -1 on period fixed effects and D g,t -D g,t -1 , among observations for which t ≥ 2 . Let β fd = E [ ̂ β fd ] .

̂ We start by showing that a result similar to Theorem 1 also applies to ̂ β fd . For any ( g, t ) ∈ { 1 , ..., G } × { 2 , ..., T } , let ε fd,g,t denote the residual of observations in group g and at period t in the regression of D g,t -D g,t -1 on period fixed effects, among observations for which t ≥ 2 . For any g ∈ { 1 , ..., G } , let ε fd,g, 1 = ε fd,g,T +1 = 0 . One can show that if the regressors in Regression 2 are not perfectly collinear,

When T = 2 and N g, 2 /N g, 1 does not vary across g , meaning that all groups experience the same growth of their number of units from period 1 to 2, one can show that ̂ β fe = ̂ β fd . ̂ β fe differs from β fd if T &gt; 2 or N g, 2 /N g, 1 varies across g .

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we define

Theorem 2 Suppose that Assumptions 1-5 hold. Then,

<!-- formula-not-decoded -->

Theorem 2 shows that under Assumption 5, β fd is equal to a weighted sum of the ATEs in each treated ( g, t ) cell with potentially some strictly negative weights, just as β fe . We now characterize the ( g, t ) cells whose ATEs are weighted negatively by β fd . To do so, we focus on

staggered adoption designs, as outside of this case it is more difficult to characterize those cells. Our characterization relies on the fact that for every t ∈ { 2 , ..., T } , ε fd,g,t = D g,t -D g,t -1 -( D .,t -D .,t -1 ) . ε fd,g,t is the difference between the change of the treatment in group g between t -1 and t , and the average change of the treatment across all groups.

Proposition 2 Suppose that Assumptions 1-2 and 6 hold and for all g , N g,t does not depend on t . Then, for all ( g, t ) such that D g,t = 1 , w fd,g,t &lt; 0 if and only if D g,t -1 = 1 and D .,t -D .,t -1 &gt; D .,t +1 -D .,t (with the convention that D .,T +1 = D .,T ).

Proposition 2 shows that for all t ∈ { 2 , ..., T -1 } such that the increase in the proportion of treated units is larger from t -1 to t than from t to t +1 , the periodt ATE of groups already treated in t -1 receives a negative weight. Moreover, if, at period T , at least one group becomes treated, the ATE of groups already treated in T -1 also receives a negative weight. Therefore, the treatment effect arising at the date when a group starts receiving the treatment does not receive a negative weight, only long-run treatment effects do. Then, negative weights are a concern when instantaneous and long-run treatment effects may differ. Proposition 2 also shows that the prevalence of negative weights depends on how the number of groups that start receiving the treatment at date t evolves with t . Assume for instance that this number decreases with t : many groups start receiving the treatment at date 1, a bit less start at date 2, etc., a case hereafter referred to as the 'more early adopters' case. Then, if N g,t is constant across ( g, t ) , D .,t -D .,t -1 is decreasing in t , and all the long-run treatment effects receive negative weights, except maybe those of period T if D .,T = D .,T -1 . Conversely, assume that the number of groups that start receiving the treatment at date t increases with t : few groups start receiving the treatment at date 1, a bit more start at date 2, etc., a case hereafter referred to as the 'more late adopters' case. Then, if N g,t is constant across ( g, t ) , D .,t -D .,t -1 is increasing in t , and only the periodT long-run treatment effects receive negative weights. Overall, negative weights are much more prevalent in the 'more early adopters' than in the 'more late adopters' case.

We now come back to general sharp designs where the treatment may not follow a staggered adoption. Let ˜ β fd = E ( ̂ β fd ∣ ∣ ∣ D ) denote the expectation of ̂ β fd conditional on the vector of treatment assignments D . Just as for ˜ β fe , one can show that the minimal value of σ ( ˜ ∆ ) compatible with ˜ β fd and ˜ ∆ TR = 0 is σ fd = | ˜ β fd | /σ ( w fd ) , where is the standard deviation of the w fd -weights. One can also show that σ fd , the minimal value of σ ( ˜ ∆ ) compatible with ˜ β fd and ˜ ∆ g,t of a different sign than ˜ β fd for all ( g, t ) , has the same expression as σ fe , except that one needs to replace the weights w g,t by the weights w fd,g,t in its

<!-- formula-not-decoded -->

definition. Estimators of σ fe and σ fd (or σ fe and σ fd ) can then be used to determine which of β fe or β fd is more robust to heterogeneous treatment effects.

<!-- formula-not-decoded -->

̂ ̂ Finally, and similarly to the result shown in Corollary 2 for β fe , β fd is equal to δ TR under common trends and the following assumption:

Note that under the common trends assumption, one can jointly test Assumption 8 and Assumption 7, the assumption that the weights attached to β fe are uncorrelated with the ∆ g,t s: if ̂ β fe and ̂ β fd are significantly different, at least one of these two assumptions must fail. In the second application we revisit in Section 6, ̂ β fe and ̂ β fd are significantly different.

## 4 An alternative estimator

In this section, we show that it is possible to estimate a well-defined causal effect even if treatment effects are heterogeneous across groups or over time. Let glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

with N S = ∑ ( g,t ): t ≥ 2 ,D g,t = D g,t -1 N g,t . δ S is the ATE of all switching cells. In staggered adoption designs, δ S is the average of the treatment effect at the time when a group starts receiving the treatment, across all groups that become treated at some point.

We now show that δ S can be unbiasedly estimated by a weighted average of DID estimators. This result holds under the following supplementary assumptions.

<!-- formula-not-decoded -->

Assumption 9 is the equivalent of Assumption 4, for the potential outcome with treatment. It requires that the shocks affecting a group's Y g,t (1) be mean independent of that group's treatment sequence.

Assumption 10 (Common trends for Y (1) ) For t ≥ 2 , E ( Y g,t (1) -Y g,t -1 (1)) does not vary across g .

Again, Assumption 10 is the equivalent of Assumption 5, for the potential outcome with treatment. It requires that between each pair of consecutive periods, the expectation of the outcome with treatment follow the same evolution over time in every group. Assumptions 9 and 10 ensure that one can reconstruct the potential outcome that groups leaving the treament between

t -1 and t would have experienced if they had remained treated. In staggered adoption designs, Assumption 9 and 10 are not necessary for identification, because no group leaves the treatment. Together, Assumptions 5 and 10 imply that the ATE follows the same evolution over time in every group: E (∆ g,t ) = η t + θ g . 13 This still allows for heterogeneous treatment effects across groups and over time. 14

Assumption 11 (Existence of 'stable' groups) For all t ∈ { 2 , ..., T } :

1. If there is at least one g ∈ { 1 , ..., G } such that D g,t -1 = 0 and D g,t = 1 , then there exists at least one g ′ = g, g ′ ∈ { 1 , ..., G } such that D g ′ ,t -1 = D g ′ ,t = 0 .

glyph[negationslash]

2. If there is at least one g ∈ { 1 , ..., G } such that D g,t -1 = 1 , D g,t = 0 , then there exists at least one g ′ = g, g ′ ∈ { 1 , ..., G } such that D g ′ ,t -1 = D g ′ ,t = 1 .

glyph[negationslash]

The first point of the stable groups assumption requires that between each pair of consecutive time periods, if there is a 'joiner' (i.e., a group switching from being untreated to treated), then there should be another group that is untreated at both dates. The second point requires that between each pair of consecutive time periods, if there is a 'leaver' (i.e., a group switching from being treated to untreated), then there should be another group that is treated at both dates.

Notice that under Assumption 11, groups' treatments are not independent, so Assumption 3 cannot hold. Accordingly, we replace Assumption 3 by Assumption 12 below. Assumption 12 requires that conditional on its own treatments, a group's outcomes be mean independent of the other groups' treatments. It is weaker than Assumption 3. Assumption 11 is necessary to show that our estimator is unbiased, but it is not necessary to show that it is consistent. Accordingly, in Section ?? of the Web Appendix, we show that our estimator is consistent under Assumption 3. For every g ∈ { 1 , ..., G } , let D g = ( D 1 ,g , ..., D T,g ) .

Assumption 12 (Mean independence between a group's outcome and other groups treatments) For all g and t , E ( Y g,t (0) | D ) = E ( Y g,t (0) | D g ) and E ( Y g,t (1) | D ) = E ( Y g,t (1) | D g ) .

We can now define our estimator. For all t ∈ { 2 , ..., T } and for all ( d, d ′ ) ∈ { 0 , 1 } 2 , let

<!-- formula-not-decoded -->

13 It should be possible to weaken Assumptions 9-10, in particular to account for dynamic effects where ∆ g,t may depend on ( D g, 1 , ..., D g,t -1 ) . This would however introduce complications that are beyond the scope of this paper.

14 Imposing Assumptions 9 and 10 does not change the decompositions obtained in Theorems 1 and 2. Y g,t (1) is observed for all the treated ( g, t ) cells entering these decompositions, so those assumptions do not bring identifying information for those cells.

denote the number of observations with treatment d ′ at period t -1 and d at period t . Let

<!-- formula-not-decoded -->

Note that DID + ,t is not defined when there is no group such that D g,t = 1 , D g,t -1 = 0 , or no group such that D g,t = 0 , D g,t -1 = 0 . In such instances, we let DID + ,t = 0 . Similarly, let DID -,t = 0 when there is no group such that D g,t = 1 , D g,t -1 = 1 or no group such that D g,t = 0 , D g,t -1 = 1 . Finally, let

<!-- formula-not-decoded -->

Theorem 3 If Assumptions 1, 2, 4, 5, and 9-12 hold, E [ DID M ] = δ S .

In Section ?? of the Web Appendix, we also show that when G goes to infinity, DID M is a consistent and asymptotically normal estimator of δ S . The DID M estimator is computed by the fuzzydid and did\_multiplegt Stata packages.

Here is the intuition underlying Theorem 3. DID + ,t compares the evolution of the mean outcome between t -1 and t in two sets of groups: the joiners, and those remaining untreated. Under Assumptions 4 and 5, DID + ,t estimates the joiners' treatment effect. Similarly, DID -,t compares the evolution of the outcome between t -1 and t in two sets of groups: those remaining treated, and the leavers. Under Assumptions 9 and 10, it estimates the leavers' treatment effect. Finally, DID M is a weighted average of those DIDs. Note that in staggered designs, there are no groups whose treatment decreases over time, so DID M is only a weighted average of the DID + ,t estimators. Note also that one can separately estimate the joiners' and the leavers' treatment effect, by computing separately weighted averages of the DID + ,t and DID -,t estimators. The former estimator only relies on Assumptions 4 and 5, while the latter only relies on Assumptions 9 and 10.

DID M is related to two other estimators. First, it is related to the Wald-TC estimator in point 2 of Theorem S1 in the Web Appendix of de Chaisemartin and D'Haultfœuille (2018), but the weighting of DID + ,t and DID -,t therein differs. As a result, DID M estimates ∆ S under weaker assumptions. DID M is also related to the multi-period DID estimator in Imai and Kim (2018). However, the multi-period DID estimator is a weighted average of the DID + ,t , so it does not estimate the leavers' treatment effect, and applies to a smaller population. Besides, Imai and Kim (2018) do not establish the properties of their estimator. Finally, they do not generalize it to non-binary treatments, something we do in Section ?? of the Web Appendix.

There may be a bias-variance trade-off between DID M and the two-way fixed effects regression estimators. For instance, assume that Regression 1 is correctly specified:

<!-- formula-not-decoded -->

Then, if the errors ε g,t are homoskedastic and uncorrelated, it follows from the Gauss-Markov theorem that ̂ β fe is the linear estimator of δ , the constant treatment effect parameter, with the lowest variance. As DID M is also an unbiased linear estimator of δ , the variance of ̂ β fe must be lower than that of DID M . With heteroskedastic or correlated errors, one can construct examples where the variance of ̂ β fe is higher than that of DID M , but this still suggests that DID M may often have a larger variance than that of β fe , as we find in our applications in Section 6.

̂ DID M uses groups whose treatment is stable to infer the trends that would have affected switchers if their treatment had not changed. This strategy could fail, if switchers experience different trends than groups whose treatment is stable. To assess if this is a serious concern, we propose to use the following placebo estimator, that essentially compares the outcome's evolution from t -2 to t -1 , in groups that switch and do not switch treatment between t -1 and t . This placebo estimator is defined under a modified version of Assumption 11.

Assumption 13 (Existence of 'stable' groups for the placebo test) For all t ∈ { 3 , ..., T } :

1. If there is at least one g ∈ { 1 , ..., G } such that D g,t -2 = D g,t -1 = 0 and D g,t = 1 , then there exists at least one g ′ = g, g ′ ∈ { 1 , ..., G } such that D g ′ ,t -2 = D g ′ ,t -1 = D g ′ ,t = 0 .

glyph[negationslash]

2. If there is at least one g ∈ { 1 , ..., G } such that D g,t -2 = D g,t -1 = 1 , D g,t = 0 , then there exists at least one g ′ = g, g ′ ∈ { 1 , ..., G } such that D g ′ ,t -2 = D g ′ ,t -1 = D g ′ ,t = 1 .

glyph[negationslash]

For all t ∈ { 2 , ..., T } and for all ( d, d ′ , d ′′ ) ∈ { 0 , 1 } 3 , let

<!-- formula-not-decoded -->

denote the number of observations with treatment status d ′′ at period t -2 , d ′ at period t -1 , and d at period t . Let glyph[negationslash]

<!-- formula-not-decoded -->

When there is no group such that D g,t = 1 , D g,t -1 = D g,t -2 = 0 or no group such that D g,t = D g,t -1 = D g,t -2 = 0 , we let DID pl + ,t = 0 , and we adopt the same convention for DID pl -,t = 0 . Let

<!-- formula-not-decoded -->

Theorem 4 If Assumptions 1, 2, 4, 5, 9, 10, 12, and 13 hold, then E [ DID pl M ] = 0 .

DID pl + ,t compares the evolution of the mean outcome from t -2 to t -1 in two sets of groups: those untreated at t -2 and t -1 but treated at t , and those untreated at t -2 , t -1 , and t . If Assumptions 4 and 5 hold, then E [ DID pl + ,t ] = 0 . Similarly, if Assumptions 9 and 10 hold, E [ DID pl -,t ] = 0 . Then, E [ DID pl M ] = 0 is a testable implication of Assumptions 4, 5, 9, and 10, so finding DID pl M significantly different from 0 would imply that those assumptions are violated: groups that switch treatment experience different trends before that switch than the groups used to reconstruct their counterfactual trends when they switch. 15 Note that DID pl M compares the trends of switching and stable groups one period before the switch. One can define other placebo estimators comparing those trends, say, two or three periods before the switch. DID pl M and all those other placebo estimators are computed by the did\_multiplegt Stata package.

## 5 Extensions

In this section, we briefly review some of the extensions in our Web Appendix. First, we show that the decomposition of β fe in Theorem 1 can be extended to fuzzy designs where the treatment varies within ( g, t ) cells, to applications with a non binary treatment, and to two-way fixed effects regressions with control variables. 16 In fuzzy designs or with a non-binary treatment, the weights in Theorem 1 remain essentially unchanged.

We also consider two-way fixed effects regressions with covariates. Specifically, we study the coefficient of D g,t in a regression of Y i,g,t on group and period fixed effects, D g,t , and a vector of covariates X g,t . We show that a result very similar to Theorem 1 applies to that coefficient, up to two differences. First, including covariates allows for different trends across groups, provided those differential trends are fully accounted for by a linear model in X g,t -X g,t -1 , the change in a group's covariates. Specifically, instead of Assumptions 4 and 5, one needs to assume that

<!-- formula-not-decoded -->

15 See also Callaway and Sant'Anna (2018), who propose another placebo test in staggered adoption designs. 16 The decomposition of β fd in Theorem 2 can also be extended to all of those cases.

for some vector γ and constant λ t , and where X g = ( X g, 1 , ..., X g,T ) . Importantly, when the covariates are group-specific linear trends, the equation above is equivalent to

<!-- formula-not-decoded -->

meaning that from t -1 to t , the evolution of Y (0) in group g should deviate from its groupspecific linear trend γ g by an amount λ t common to all groups. Second, the residual ε g,t in the weights in Theorem 1 has to be replaced by ε X g,t , the residual of observations in cell ( g, t ) in the regression of D g,t on group and period fixed effects and X g,t . Some of the corresponding weights may still be negative, as in Theorem 1. Overall, two-way fixed effects regressions with covariates may rely on a more plausible common trends assumptions than those without covariates, but they still require that the treatment effect be homogeneous, across time and between groups.

Third, we show that under the common trends assumption and the assumption that the ATE of a ( g, t ) cell does not change over time, β fe and β fd identify weighted sums of the ATEs of the ( g, t ) cells whose treatment changes between t -1 and t . In sharp designs, the weights attached to β fd are all positive, while for β fe , the same only holds in staggered adoption designs.

Fourth, we show that our DID M estimator can easily be extended to non-binary, discrete treatments. Then, we define it as a weighted average of DIDs comparing the evolution of the outcome in groups whose treatment went from d to d ′ between t -1 and t and in groups with a treatment of d at both dates, across all possible values of d , d ′ , and t .

Finally, our twowayfeweights , fuzzydid , and did\_multiplegt Stata packages can handle all of those extensions.

## 6 Applicability, and applications

## 6.1 Applicability

We conducted a review of all papers published in the American Economic Review (AER) between 2010 and 2012 to assess the importance of two-way fixed effects regressions in economics. Over these three years, the AER published 337 papers. Out of these 337 papers, 33 or 9.8% of them estimate the FE or FD Regression, or other regressions resembling closely those regressions. When one withdraws from the denominator theory papers and lab experiments, the proportion of papers using these regressions raises to 19.1%.

Table 1: Papers using two-way fixed effects regressions published in the AER

|                                                  | 2010   | 2011   | 2012   | Total   |
|--------------------------------------------------|--------|--------|--------|---------|
| Papers using two-way fixed effects regressions   | 5      | 14     | 14     | 33      |
| % of published papers                            | 5.2%   | 12.2%  | 11.2%  | 9.8%    |
| % of empirical papers, excluding lab experiments | 12.8%  | 23.0%  | 19.2%  | 19.1%   |

Notes . This table reports the number of papers using two-way fixed effects regressions published in the AER from 2010 to 2012.

Table 2 shows descriptive statistics about the 33 2010-2012 AER papers estimating two-way fixed effects regressions. Panel A shows that 13 use the FE regression; six use the FD regression; six use regressions the FE or FD regression with several treatment variables; three use the FE or FD 2SLS regression discussed in Section ?? of the Web Appendix; five use other regressions that we deemed sufficiently close to the FE or FD regression to include them in our count. 17 Panel B shows that more than three fourths of those papers consider sharp designs, while less than one fourth consider fuzzy designs. Finally, Panel C assesses whether, in those applications, there are groups whose exposure to the treatment remains stable between each pair of consecutive time periods, the condition that has to be met to be able to compute the DID M estimator. For about a half of the papers, reading the paper was not enough to assess this with certainty. We then assessed whether they presumably have stable groups or not. Overall, 12 papers have stable groups, 14 presumably have stable groups, five presumably do not have stable groups, and two do not have stable groups.

In Section ?? of the Web Appendix, we review each of the 33 papers. We explain where two-way fixed effects regressions are used in the paper, and we detail our assessment of whether the design is a sharp or a fuzzy design, and of whether the stable groups assumption holds or not.

17 For instance, two papers use regressions with three-way fixed-effects instead of two-way fixed effects.

Table 2: Descriptive statistics on two-way fixed effects papers

|                                                                                    | # Papers   |
|------------------------------------------------------------------------------------|------------|
| Panel A. Estimation method                                                         |            |
| Fixed-effects OLS regression                                                       | 13         |
| First-difference OLS regression                                                    | 6          |
| Fixed-effects or first-difference OLS regression, with several treatment variables | 6          |
| Fixed-effects or first-difference 2LS regression                                   | 3          |
| Other regression                                                                   | 5          |
| Panel B. Research design                                                           |            |
| Sharp design                                                                       | 26         |
| Fuzzy design                                                                       | 7          |
| Panel C. Are there stable groups?                                                  |            |
| Yes                                                                                | 12         |
| Presumably yes                                                                     | 14         |
| Presumably no                                                                      | 5          |
| No                                                                                 | 2          |

Notes . This table reports the estimation method and the research design used in the 33 papers using two-way fixed effects regressions published in the AER from 2010 to 2012, and whether those papers have stable groups.

## 6.2 Application to Gentzkow et al. (2011)

Gentzkow et al. (2011) study the effect of newspapers on voters' turnout in US presidential elections between 1868 and 1928. They regress the first-difference of the turnout rate in county g between election years t -1 and t on state-year fixed effects and on the first difference of the number of newspapers available in that county. This corresponds to Regression 2, with state-year fixed effects as controls. As reproduced in Table 3 below, Gentzkow et al. (2011) find that ̂ β fd = 0 . 0026 (s.e.= 9 × 10 -4 ). According to this regression, one more newspaper increased voters' turnout by 0.26 percentage points. On the other hand, ̂ β fe = -0 . 0011 (s.e.= 0 . 0011 ). β fe and β fd are significantly different (t-stat=2.86).

̂ ̂ We use the twowayfeweights Stata package, downloadable with its help file from the SSC repository, to estimate the weights attached to ̂ β fe . 6,212 are strictly positive, 4,161 are strictly negative. The negative weights sum to -0.53. ̂ σ fe = 3 × 10 -4 , meaning that β fe and the ATT may be of opposite signs if the standard deviation of the ATEs across all the treated ( g, t ) cells

is equal to 0 . 0003 . 18 ̂ σ fe = 7 × 10 -4 , meaning that β fe may be of a different sign than the ATEs of all the treated ( g, t ) cells if the standard deviation of those ATEs is equal to 0 . 0007 . We also estimate the weights attached to ̂ β fd . 5,472 are strictly positive, and 4,605 are strictly negative. The negative weights sum to -1.43. σ fd = 4 × 10 -4 , and σ fd = 6 × 10 -4 .

̂ ̂ Therefore, β fe and β fd can only receive a causal interpretation if the weights attached to them are uncorrelated with the intensity of the treatment effect in each county × election-year cell (Assumptions 7 and 8, respectively). This is not warranted. First, as ̂ β fe and ̂ β fd significantly differ, Assumptions 7 and 8 cannot jointly hold. Moreover, the weights attached to ̂ β fe and ̂ β fd are correlated with variables that are likely to be themselves associated with the intensity of the treatment effect in each cell. For instance, the correlation between the weights attached to ̂ β fd and t , the year variable, is equal to -0 . 06 (t-stat=-3.28). The effect of newspapers may be different in the last than in the first years of the panel. For instance, new means of communication, like the radio, appear in the end of the period under consideration, and may diminish the effect of newspapers. 19 This would lead to a violation of Assumption 8.

The stable groups assumption holds: between each pair of consecutive elections, there are counties where the number of newspapers does not change. We use the fuzzydid Stata package, downloadable with its help file from the SSC repository, to estimate a modified version of our DID M estimator, that accounts for the fact that the number of newspapers is not binary (see section ?? of our Web Appendix, where we define this modified estimator). We include stateyear fixed effects as controls in our estimation. We find that DID M = 0 . 0043 , with a standard error of 0 . 0015 . DID M is 66% larger than ̂ β fd , and the two estimators are significantly different at the 10% level (t-stat=1.69). DID M is also of a different sign than β fe .

̂ Our DID M estimator only relies on a common trends assumption. To assess its plausibility, we compute DID pl M , the placebo estimator introduced in Section 4. 20 As shown in Table 3 below, our placebo estimator is small and not significantly different from 0, meaning that counties where the number of newspapers increased or decreased between t -1 and t did not experience significantly different trends in turnout from t -2 to t -1 than counties where that number was stable. Our placebo estimator is estimated on a subset of the data: for each pair of consecutive time periods t -1 and t , we only keep counties where the number of newspapers did not change between t -2 and t -1 . Still, almost 80% of the county × election-year observations are used in the computation of the placebo estimator. Moreover, when reestimated on this subsample, the DID M estimator is very close to the DID M estimator in the full sample.

18 The number of newspapers is not binary, so strictly speaking, in this application the parameter of interest is the average causal response parameter introduced in Section ?? of our Web Appendix, rather than the ATT.

20 Again, we need to slightly modify DID pl M to account for the fact that the number of newspapers is not binary.

19 In fact, Gentzkow et al. (2011) analyze the 1868 to 1928 period separately from later periods, because the growth of the radio may have changed newspapers' effects.

Table 3: Estimates of the effect of one additional newspaper on turnout

|                              |   Estimate |   Standard error | N      |
|------------------------------|------------|------------------|--------|
| ̂ β fd                        |     0.0026 |           0.0009 | 15,627 |
| β fe                         |    -0.0011 |           0.0011 | 16,872 |
| ̂ DID M                       |     0.0043 |           0.0015 | 16,872 |
| DID pl M                     |    -0.0009 |           0.0016 | 13,221 |
| DID M , on placebo subsample |     0.0045 |           0.0019 | 13,221 |

Notes . This table reports estimates of the effect of one additional newspaper on turnout, as well as a placebo estimate of the common trends assumption underlying DID M . Estimators are computed using the data of Gentzkow et al. (2011), with state-year fixed effects as controls. Standard errors are clustered by county. To compute the DID M estimators, the number of newspapers is grouped into 4 categories: 0, 1, 2, and more than 3.

## 6.3 The effect of union membership on wages

A number of articles have estimated the effect of union membership on wages using panel data and controlling for workers' fixed effects. For instance, Jakubson (1991) has found a 8.3% union membership premium using that strategy, in a sample of American males from the PSID followed from 1976 to 1980. Vella and Verbeek (1998) estimate a similar regression and find similar results, in a sample of young American males from the NLSY followed from 1980 to 1987. 21

We use the data in Vella and Verbeek (1998) to compute various estimators of the union wage premium. As union status is often measured with error (see, e.g. Freeman, 1984; Card, 1996), we discard changes in union status happening twice in three consecutive years. Specifically, for individuals with D i,t -1 = 0 , D i,t = 1 , and D i,t +1 = 0 , we replace D i,t by 0. Similarly, for individuals with D i,t -1 = 1 , D i,t = 0 , and D i,t +1 = 1 , we replace D i,t by 1. Doing so, we discard half of the union status changes in the initial data. 22

We start by estimating a two-way fixed effects regression of wages on union membership with worker and year fixed effects. Table 4 below shows that ̂ β fe = 0 . 107 (s.e.= 0 . 030 ), a result close to that of the worker fixed effects regressions in Jakubson (1991) and Vella and Verbeek (1998).

21 The fixed effects regression is not the main specification in Vella and Verbeek (1998). The authors favor instead a dynamic selection model.

Then, we estimate the weights attached to ̂ β fe . 820 are strictly positive, 196 are strictly negative, but the negative weights only sum to -0.01. Still, ̂ σ fe = 0 . 097 , meaning that β fe and the ATT may be of opposite signs if the standard deviation of the treatment effect across the unionized worker × year observations is equal to 0 . 097 , a substantial but still possible amount of heterogeneity.

22 Keeping the original data does not change much the results presented below, except that the placebo estimator DID pl , 2 M becomes significant.

The weights are negatively correlated with workers' years of schooling (correlation = -0 . 12 , t-stat = -1 . 88 ). The union premium may be lower for more educated workers (see Freeman and Medoff, 1984), as they may be less substitutable than less educated ones. Then, ̂ β fe may overestimate δ TR , the average union premium across all unionized worker × year observations. We also find that ̂ β fd = 0 . 060 (s.e.= 0 . 032 ) and that ̂ β fe and ̂ β fd significantly differ (t-stat=1.91), 23 thus casting further doubt on Assumptions 7 and 8.

The stable groups assumption holds: between each pair of consecutive years, there are workers whose union membership status does not change. We therefore compute our DID M estimator. Table 4 shows that it is equal to 0 . 041 (s.e. = 0 . 034 ). DID M is significantly different from ̂ β fe (t-stat=2.60) and ̂ β fd (t-stat=2.36). 24 As discussed in Section 4, we can also estimate separately the union premium for workers joining and leaving a union, something that was previously done by Freeman (1984). The joiners' effect estimate is equal to 0 . 059 (s.e. = 0 . 053 ), the leavers' effect is equal to 0 . 021 (s.e. = 0 . 044 ), and the two estimates do not significantly differ (t-stat = 0 . 55 ).

DID M relies on a common trends assumption. To assess its plausibility, we compute DID pl M , the placebo estimator introduced in Section 4. DID pl M compares the wage growth of workers changing and not changing their union status one period before that change. We also compute DID pl , 2 M and DID pl , 3 M , two other placebo estimators performing the same comparison two and three periods before the change. As shown in Table 4 below, DID pl M is large, positive, and significant (t-stat=2.49). On the other hand DID pl , 2 M and DID pl , 3 M are smaller and insignificant. Workers that become unionized start experiencing a differential positive pre-trend one year before becoming unionized. This differential pre-trend mostly comes from union joiners: for them, the placebo estimator is equal to 0 . 119 (s.e. = 0 . 051 ), while for union leavers the placebo is smaller ( 0 . 061 ) and insignificant (s.e. = 0 . 057 ). Therefore, the placebos suggest that even the already small and insignificant DID M estimator may overestimate the union premium, due to a positive pre-trend. In fact, the estimate of leavers' effect, for which there is no evidence of a pre-trend, is very close to 0. Overall, our results indicate that there may not be a significant union wage premium.

23 The standard error of ̂ β fe -̂ β fd is computed with a worker-level clustered bootstrap.

24 The standard errors of ̂ β fe -DID M and ̂ β fd -DID M are computed with a worker-level clustered bootstrap.

Table 4: Estimates of the union premium

|              |   Estimate |   Standard error | N     |
|--------------|------------|------------------|-------|
| β fe         |      0.107 |            0.03  | 4,360 |
| ̂ β fd        |      0.06  |            0.032 | 3,815 |
| ̂ DID M       |      0.041 |            0.034 | 3,815 |
| DID pl M     |      0.094 |            0.038 | 3,101 |
| DID pl , 2 M |     -0.041 |            0.03  | 2,458 |
| DID pl , 3 M |     -0.004 |            0.033 | 1,881 |

Notes . This table reports estimates of the effect of the union premium, as well as placebo estimators of the common trends assumption. Estimators are computed using the data of Vella and Verbeek (1998). Standard errors are clustered at the worker level.

## 7 Conclusion

Almost 20% of empirical articles published in the AER between 2010 and 2012 use regressions with groups and period fixed effects to estimate treatment effects. In this paper, we show that under a common trends assumption, those regressions estimate weighted sums of the treatment effect in each group and period. The weights may be negative: in one application, we find that almost 50% of the weights are negative. The negative weights are an issue when the treatment effect is heterogeneous, between groups or over time. Then, one could have that the treatment's coefficient in those regressions is negative while the treatment effect is positive in every group and time period. We therefore propose a new estimator to address this problem. This estimator estimates the treatment effect in the groups that switch treatment, at the time when they switch. It does not rely on any treatment effect homogeneity condition. It is computed by the fuzzydid and did\_multiplegt Stata packages. In the two applications we revisit, this estimator is significantly and economically different from the two-way fixed effects estimators.

## References

- Abraham, S. and Sun, L. (2018), Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Working Paper.
- Ashenfelter, O. (1978), 'Estimating the effect of training programs on earnings', The Review of Economics and Statistics pp. 47-57.
- Athey, S. and Imbens, G. W. (2018), Design-based analysis in difference-in-differences settings with staggered adoption, Technical report, National Bureau of Economic Research.
- Athey, S. and Stern, S. (2002), 'The impact of information technology on emergency health care outcomes', The RAND Journal of Economics 33 (3), 399-432.
- Autor, D. H. (2003), 'Outsourcing at will: The contribution of unjust dismissal doctrine to the growth of employment outsourcing', Journal of labor economics 21 (1), 1-42.
- Borusyak, K. and Jaravel, X. (2017), Revisiting event study designs. Working Paper.
- Callaway, B. and Sant'Anna, P. H. (2018), Difference-in-differences with multiple time periods and an application on the minimum wage and employment. arXiv e-print 1803.09015.
- Card, D. (1996), 'The effect of unions on the structure of wages: A longitudinal analysis', Econometrica: Journal of the Econometric Society pp. 957-979.
- de Chaisemartin, C. (2011), Fuzzy differences in differences. Working Paper 2011-10, Center for Research in Economics and Statistics.
- de Chaisemartin, C. and D'Haultfoeuille, X. (2015), Fuzzy differences-in-differences. ArXiv e-prints, eprint 1510.01757v2.
- de Chaisemartin, C. and D'Haultfœuille, X. (2018), 'Fuzzy differences-in-differences', The Review of Economic Studies 85 (2), 999-1028.
- de Chaisemartin, C., D'Haultfœuille, X. and Guyonvarch, Y. (2019), 'Fuzzy differences-indifferences with Stata', Stata Journal 19 (2), 435-458.
- Duflo, E. (2001), 'Schooling and labor market consequences of school construction in indonesia: Evidence from an unusual policy experiment', American Economic Review 91 (4), 795-813.
- Frank, M. and Wolfe, P. (1956), 'An algorithm for quadratic programming', Naval research logistics quarterly 3 (1-2), 95-110.

- Freeman, R. B. (1984), 'Longitudinal analyses of the effects of trade unions', Journal of labor Economics 2 (1), 1-26.
- Freeman, R. B. and Medoff, J. L. (1984), 'What do unions do', Indus. &amp; Lab. Rel. Rev. 38 , 244.
- Gentzkow, M., Shapiro, J. M. and Sinkinson, M. (2011), 'The effect of newspaper entry and exit on electoral politics', American Economic Review 101 (7), 2980-3018.
- Goodman-Bacon, A. (2018), Difference-in-differences with variation in treatment timing. Working Paper.
- Imai, K. and Kim, I. S. (2018), 'On the use of two-way fixed effects regression models for causal inference with panel data'.
- Jakubson, G. (1991), 'Estimation and testing of the union wage effect using panel data', The Review of Economic Studies 58 (5), 971-991.
- Vella, F. and Verbeek, M. (1998), 'Whose wages do unions raise? a dynamic model of unionism and wage rate determination for young men', Journal of Applied Econometrics 13 (2), 163-183.
- Wooldridge, J. M. (2002), Econometric analysis of cross section and panel data , MIT press.

## A Proofs

## One useful lemma

Our results rely on the following lemma.

Lemma 1 If Assumptions 1-5 hold, for all ( g, g ′ , t, t ′ ) ∈ { 1 , ..., G } 2 ×{ 1 , ..., T } 2 ,

<!-- formula-not-decoded -->

Proof of Lemma 1

For all ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } ,

<!-- formula-not-decoded -->

where the third equality follows from Assumption 2, and the fourth from Assumption 3. Therefore,

<!-- formula-not-decoded -->

where the first equality follows from Assumption 3, the second from the linearity of the conditional expectation and Assumption 4, and the third from Assumption 5.

## Proof of Theorem 1

It follows from the Frisch-Waugh theorem and the definition of ε g,t that

<!-- formula-not-decoded -->

Now, by definition of ε g,t again,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

The first and third equalities follow from Equations (5) and (6). The second equality follows from Lemma 1. The fourth equality follows from Assumption 2. Finally, Assumption 2 implies that

Combining (4), (8), (9) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the result follows from the law of iterated expectations.

## Proof of Proposition 1

If for all t ≥ 2 N g,t /N g,t -1 does not depend on t , then it follows from the first order conditions attached to Regression 1 and a few lines of algebra that ε g,t = D g,t -D g,. -D .,t + D .,. . Therefore, w g,t is proportional to D g,t -D g,. -D .,t + D .,. . Then, for all ( g, t, t ′ ) such that D g,t = D g,t ′ = 1 , D .,t &gt; D .,t ′ implies w g,t &lt; w g,t ′ . Similarly, for all ( g, g ′ , t ) such that D g,t = D g ′ ,t = 1 , D g,. &gt; D g ′ ,. implies w g,t &lt; w g ′ ,t .

## Proof of Corollary 1

## Proof of the first point

We start by proving the first point. If the assumptions of the corollary hold and ˜ ∆ TR = 0 , then where the first equality follows from (10). These two conditions and the Cauchy-Schwarz inequality imply

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we prove that we can rationalize this lower bound. Let us define

Hence, σ ( ˜ ∆ ) ≥ σ fe .

Then,

<!-- formula-not-decoded -->

as it follows from the definition of w g,t that ∑ ( g,t ): D g,t =1 N g,t N 1 w g,t = 1 . Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof of the second point

˜ where the second equality follows again from the fact that ∑ ( g,t ): D g,t =1 N g,t N 1 w g,t = 1 .

We first suppose that β fe &gt; 0 . We seek to solve:

<!-- formula-not-decoded -->

This is a quadratic programming problem, with a matrix that is symmetric positive but not definite. Hence, by Frank and Wolfe (1956) and the fact that the linear term in the quadratic problem is 0, the solution exists if and only if the set of constraints is not empty. If w ( n ) ≥ 0 , the set of constraints is empty because ∑ n i =1 N ( i ) N 1 w ( i ) ˜ ∆ ( i ) ≤ 0 &lt; ˜ β fe . On the other hand, if w ( n ) &lt; 0 , this set is non-empty since it includes (0 , ..., 0 , ˜ β fe / ( P ( n ) w ( n ) )) . We now derive the corresponding bound. For that purpose, remark that

The Karush-Kuhn-Tucker necessary conditions for optimality are that for all i :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ ˜ These constraints imply that ˜ ∆ ( i ) = 0 if and only if ˜ ∆ TR + λw ( i ) ≥ 0 . Therefore, if ˜ ∆ TR + λw ( i ) &lt; 0 , ˜ ∆ ( i ) = 0 so γ ( i ) = 0 , and ˜ ∆ ( i ) = ˜ ∆ TR + λw ( i ) . Therefore,

˜ where ˜ ∆ TR = ∑ n i =1 N ( i ) N 1 ˜ ∆ ( i ) , 2 λ is the Lagrange multiplier of the constraint ∑ n i =1 N ( i ) N 1 w ( i ) ˜ ∆ ( i ) = β fe and 2 N ( i ) N 1 γ ( i ) is the Lagrange multiplier of the constraint ∆ ( i ) ≤ 0 .

glyph[negationslash]

This equation implies that ˜ ∆ ( i ) ≤ ˜ ∆ TR + λw ( i ) , which in turn implies that ˜ ∆ TR ≤ ˜ ∆ TR + λ , so λ ≥ 0 .

<!-- formula-not-decoded -->

As a result, ˜ ∆ TR + λw ( i ) is decreasing in i , and because x ↦→ min( x, 0) is increasing, ˜ ∆ ( i ) is also decreasing in i . Then ˜ ∆ ( n ) &lt; 0 : otherwise one would have ˜ ∆ ( i ) = 0 for all i which would imply ˜ β fe = 0 , a contradiction. Let s = min { i ∈ { 1 , ..., n } : ∆ ( i ) &lt; 0 } . Using again (11), we get

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, plugging ˜ ∆ in (11), we obtain that for all i ≥ s ,

Finally, using again (11), we obtain

Thus,

Then, using what precedes,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, consider the case ˜ β fe &lt; 0 . By letting ˜ ∆ ′ ( i ) = -˜ ∆ ( i ) and ˜ β ′ fe = -˜ β fe , we have

The result follows, once noted that Equations (11) and (12) imply that s = min { i ∈ { 1 , ..., n } : w ( i ) &lt; -S ( i ) / (1 -P ( i ) ) } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is the same program as before, with ˜ β ′ fe instead of ˜ β fe . Therefore, by the same reasoning as before, we obtain

## Proof of Corollary 2

We have

The first equality follows from the law of iterated expectations and (10). The second equality follows from Assumption 7. By the definition of w g,t , ∑ ( g,t ): D g,t =1 N g,t N 1 w g,t = 1 , hence the third equality. The fourth equality follows from the law of iterated expectations.

## Proof of Theorem 2

It follows from the Frisch-Waugh theorem and the definition of ε fd,g,t that

<!-- formula-not-decoded -->

Now, by definition of ε fd,g,t again,

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

The first and third equalities follow from (14). The second equality follows from Lemma 1. The fourth equality follows from a summation by part, and from the fact ε fd,g, 1 = ε fd,g,T +1 = 0 . The fifth equality follows from Assumption 2.

A similar reasoning yields

<!-- formula-not-decoded -->

Combining (13), (15), (16), and the law of iterated expectations yields the result.

## Proof of Proposition 2

It follows from the first order conditions attached to Regression 2 and a few lines of algebra that ε fd,g,t = D g,t -D g,t -1 -D .,t + D .,t -1 . Therefore, under Assumption 6 and if N g,t does not vary across t , one has that for all ( g, t ) such that D g,t = 1 , 1 ≤ t ≤ T -1 , w fd,g,t is proportional to 1 -D g,t -1 -(2 D .,t -D .,t -1 -D .,t +1 ) . D .,t -D .,t -1 ≤ 1 , and under Assumption 6 D .,t -D .,t +1 ≤ 0 , so 1 -D g,t -1 -(2 D .,t -D .,t -1 -D .,t +1 ) can only be strictly negative if D g,t -1 = 1 . Then, for all ( g, t ) such that D g,t = 1 , 1 ≤ t ≤ T -1 , w fd,g,t is strictly negative if and only if D g,t -1 = 1 and 2 D .,t -D .,t -1 -D .,t +1 &gt; 0 .

Similarly, when t = T , under the same assumptions as above, one has that for all g such that D g,T = 1 , w fd,g,T is proportional to 1 -D g,T -1 -( D .,T -D .,T -1 ) . D .,T -D .,T -1 ≤ 1 , so 1 -D g,T -1 -( D .,T -D .,T -1 ) can only be strictly negative if D g,T -1 = 1 . Then, w fd,g,T is strictly negative if and only if D g,T -1 = 1 and D .,T -D .,T -1 &gt; 0 .

Finally, when t = 1 , one has that for all g such that D g, 1 = 1 , D g, 2 = 1 under Assumption 6, so w fd,g, 1 is proportional to D ., 2 -D ., 1 , which is greater than 0 under Assumption 6.

## Proof of Theorem 3

First, by definition of DID M ,

<!-- formula-not-decoded -->

Let t be greater than 2, and let us focus for now on the case where there is at least one g 1 such that D g 1 ,t -1 = 0 and D g 1 ,t = 1 . Then Assumption 11 ensures that there is a least another group g 2 such that D g 2 ,t -1 = D g 2 ,t = 0 . For every g such that D g,t -1 = 0 and D g,t = 1 , we have

<!-- formula-not-decoded -->

Under Assumptions 12, 4, and 5, for all t ≥ 2 , there exists a real number ψ 0 ,t such that for all g

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

The first equality follows by (18), the second by (19), and the third after some algebra. If there is no g such that D g,t -1 = 0 and D g,t = 1 , (20) still holds, as DID + ,t = 0 in this case. A similar reasoning yields

<!-- formula-not-decoded -->

Plugging (20) and (21) into (17) yields

<!-- formula-not-decoded -->

## Proof of Theorem 4

First, as with DID M , we have

<!-- formula-not-decoded -->

Let t be greater than 3, and let us for now focus on the case where there exists at least one g 1 such that D g 1 ,t -2 = D g 1 ,t -1 = 0 and D g 1 ,t = 1 . Then Assumption 13 ensures that there is a least

another group g 2 such that D g 2 ,t -2 = D g 2 ,t -1 = D g 2 ,t = 0 . Then,

<!-- formula-not-decoded -->

The second equality follows by (19), and the third follows after some algebra. If there exists no g such that D g,t -2 = D g,t -1 = 0 and D g,t = 1 , (23) still holds, as DID pl + ,t = 0 in this case. A similar reasoning yields

The result follows after plugging (23) and (24) into (22).

<!-- formula-not-decoded -->

## Web appendix of 'Two-way fixed effects estimators with heterogeneous treatment effects'

Clément de Chaisemartin ∗

Xavier D'Haultfœuille †

February 13, 2020

## Abstract

In this web appendix, we first discuss whether common trends necessarily implies homogenous treatment effect. Second, we show that the decomposition in Theorem 1 in the paper extends to fuzzy designs, to regressions with covariates, to regressions with a non-binary treatment, and we derive another decomposition under the supplementary assumption that the treatment effect does not change over time. Third, we extend the DID M estimator to non-binary treatments. Fourth, we discuss inference. Fifth, we review all the papers included in our survey of papers published in the AER between 2010 and 2012 (see Section 6 of the paper). Finally, the last section gathers the proofs of all the additional results in this Web Appendix.

## 1 Can common trends hold with heterogeneous treatment effects?

Throughout the paper, we assume that groups experience common trends, but that the effect of the treatment may be heterogeneous between groups and / or over time. We now discuss two examples where this may happen. We then argue that the mechanisms behind these examples are fairly general. Thus treatment effects are often likely to be heterogeneous, even when common trends are plausible.

First, assume one wants to learn the effect of the minimum wage on the employment levels of some US counties. For simplicity, let us assume that the minimum wage can only take two values, a low and a high value. Also, let us assume that there are only two periods, the 90s and the 2000s. Between these two periods, the amount of competition from China for the US industry increased substantially. Thus, for the common trends assumption to hold for counties A and B, the effect of that increase in competition should be the same on averge in those two

∗ University of California at Santa Barbara, clementdechaisemartin@ucsb.edu

† CREST, xavier.dhaultfoeuille@ensae.fr

counties, in the counterfactual state of the world where A and B have a low minimum wage at both dates. For that to be true, the economy of those two counties should be pretty similar. For instance, if A has a very service-oriented economy, while B has a very industry-oriented economy, it is unlikely that their employment levels will react similarly to Chinese competition.

Now, if the economies of A and B are similar, they should also have similar effects of the minimum wage on employment, thus implying that the treatment effect is homogenous between groups. On the other hand, the treatment effect may vary over time. For instance, the drop in the employment levels of A and B due to Chinese competition will probably be higher if their minimum wage is high than if their minimum wage is low. This is equivalent to saying that the effect of the minimum wage on employment diminishes from the first to the second period: due to Chinese competition in the second period, the minimum wage may have a more negative effect on employment then. 1

Second, assume one wants to learn the effect of a job training program implemented in some US counties on participants' wages. Let us suppose that individuals self-select into the training according to a Roy model:

<!-- formula-not-decoded -->

where c g,t represents the cost of the training for individuals in county g and period t . We consider fuzzy designs such as this one in the next section. Here, the common trends condition requires that average wages without the training follow the same evolution in all counties. As above, for this to hold counties used in the analysis should have similar economies, so let us assume that those counties are actually identical copies of each other: at each period, their distribution of wages without and with the training is the same. Therefore, ( g, t ) ↦→ E ( Y g,t (1) -Y g,t (0)) is constant. However, c g,t may vary across counties and over time: some counties may subsidize the training more than others, and some counties may change their subsidies over time. Then, ( g, t ) ↦→ E ( Y i,g,t (1) -Y i,g,t (0) | D i,g,t = 1) will not be constant, despite the fact that all counties in the sample have similar economies and experience similar trends on their wages.

Overall, when the treatment is assigned at the group × period level as in the minimum wage example, the economic restrictions underlying the common trends assumption may also imply homogeneous treatment effect between groups. However, those restrictions usually do not imply that the treatment effect is constant over time. Moreover, when the treatment is assigned at the individual level, as in the job training example, the economic restrictions underlying the common trends assumption neither imply homogeneous treatment effects between groups, nor homogeneous treatment effects over time.

1 To simplify our discussion, in this example we consider only two counties. But in order to estimate consistently average treatment effects in the presence of county-specific shocks, the number of groups should tend to infinity, as in Section 5 below.

## 2 Results in fuzzy designs

In this section, the research design may be fuzzy: the treatment may vary within ( g, t ) cells. For instance, Enikolopov et al. (2011) study the effect of having access to an independent TV channel in Russia, and in each Russian region some people have access to that channel while other people do not.

## 2.1 Generalizing the decomposition of β fe to fuzzy designs

For any ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } , let

<!-- formula-not-decoded -->

denote the average treatment effect across the treated units of cell cell ( g, t ) . One has

<!-- formula-not-decoded -->

which generalizes (2) to fuzzy designs. 2 Theorem 1 shows that β fe is also equal to the expectation of a weighted sum of the ∆ TR g,t s. Let

<!-- formula-not-decoded -->

Theorem S1 Suppose that Assumptions 1 and 3-5 hold. Then,

<!-- formula-not-decoded -->

Theorem S1 shows that in fuzzy designs, β fe is equal to the expectation of a weighted sum of the ATTs in each ( g, t ) cell. Again, some of the weights may be strictly negative. Note that under Assumption 2, Theorem S1 reduces to Theorem 1 in the paper.

The weights have a simple expression in the following special case.

Assumption S1 (Heterogenous adoption) T = 2 and for all g ∈ { 1 , ..., G } , D g, 2 &gt; D g, 1 = 0 .

Assumption S1 is satisfied in applications with two time periods, and where all groups are fully untreated at t = 1 and partly treated at t = 2 . This type of design often arises in practice, for instance when an innovation is heterogeneously adopted by various groups.

2 Any equation with a numbering lower than (23) refers to an equation in the paper.

Proposition S1 If Assumptions 1 and S1 hold and N g, 2 /N g, 1 does not vary across g , then 3

<!-- formula-not-decoded -->

Proposition S1 shows that in the heterogeneous adoption design, β fe assigns negative weights to the period-two ATT of groups with a mean treatment lower than the mean treatment in the full population. The reason why negative weights arise is intuitive. With two periods, the FE regression is equivalent to a regression of the first difference of the outcome on the period-two treatment in each group. This regression compares the evolution of the outcome in more- and less-treated groups. Doing so, it subtracts the treatment effect of the less-treated groups, hence the negative weights. Negative weights are a concern if the ATTs of the less- and more-treated groups systematically differ. This could be the case if treatment is determined by a Roy selection model. Then, the groups with the highest proportion of treated units could also be those where the ATT is the highest. On the other hand, if the proportion of treated units is randomly assigned to each group, negative weights are not a concern. 4

The DID M estimator can also be generalized to fuzzy designs, see point 2 of Theorem S1 in the Web Appendix of de Chaisemartin and D'Haultfœuille (2018) for further details.

## 2.2 Application to Enikolopov et al. (2011)

Enikolopov et al. (2011) study the effect of NTV, an independent TV channel introduced in 1996 in Russia, on the share of the electorate voting for opposition parties. NTV's coverage rate was heterogeneous across subregions: while a large fraction of the population received NTV in urbanized subregions, a smaller fraction received it in more rural subregions. The authors estimate the FE regression: they regress the share of votes for opposition parties in the 1995 and 1999 elections in Russian subregions on subregion fixed effects, an indicator for the 1999 election, and on the share of the population having access to NTV in each subregion at the time of the election. In 1995, the share of the population having access to NTV was equal to 0 in all subregions, while in 1999 it was strictly greater than 0 everywhere. Therefore, the authors' research design corresponds exactly to the heterogenous adoption design discussed above. Enikolopov et al. (2011) find that ̂ β fe = 6 . 65 (s.e.= 1 . 40 ). According to this regression, increasing the share of the population having access to NTV from 0 to 100% increases the share of votes for the opposition parties by 6.65 percentage points. Because there are only two time periods in the data and the regression is not weighted by subregions' populations, β fe = β fd .

4 Corollaries 1 and 2 extend directly to fuzzy settings.

̂ ̂ We use the twowayfeweights Stata package, downloadable with its help file from the SSC repository, to compute the weights attached to ̂ β fe . In 1995, all the weights are equal to zero 3 Under Assumption S1, D g, 1 = 0 , so w TR g, 1 does not enter in the decomposition in Theorem 1.

because NTV does not exist yet. In 1999, 918 weights (47.4%) are strictly positive, while 1,020 (52.6%) are strictly negative. The negative weights sum to -2.26. ̂ σ fe = 0 . 91 : β fe and δ TR may be of opposite signs if the standard deviation of the effect of NTV across subregions is above 0.91 percentage point. ̂ σ fe = 1 . 23 : β fe may be of a different sign than the treatment effect in every subregion if the standard deviation of the effect of NTV across subregions is above 1.23 percentage point, a plausible amount of treatment effect heterogeneity.

Therefore, β fe can only receive a causal interpretation if the effect of NTV is constant across subregions, or if the weights attached to it are uncorrelated with the intensity of that effect in each subregion (Assumption 7). These assumptions are not warranted. First, we estimate ̂ β fe again, weighting the regression by subregions' populations. We obtain ̂ β fe = 14 . 89 , more than twice its value in the unweighted regression, and the difference between the coefficients is significant (t-stat=2.46). Therefore, we can reject the null that the treatment effect is constant: if the treatment effect was constant across subregions, the weighting would not matter so both the unweighted and the weighted regressions would estimate the same parameter. Second, the weights attached to ̂ β fe are correlated with variables that are likely to be themselves associated with the intensity of the effect in each subregion. For instance, the correlation between the weights and subregions' populations is equal to 0.35 (t-stat=14.01). The effect of NTV may be higher in less populated subregions, as those regions are more rural and fewer other sources of information may be available there. This would lead to a violation of Assumption 7.

## 3 Extensions of the decomposition results

We consider hereafter several extensions of our decompositions of β fe and β fd in the paper. First, we consider decompositions under the common trends assumption, and under the assumption that the treatment effect is stable over time. Second, we extend our decompositions to ordered treatments. Third, we investigate the effect of including covariates in the regression. Fourth, we study two-way fixed effects 2SLS regressions. Throughout, we focus on sharp designs to ease the exposition. Nevertheless, all the results generalize to fuzzy designs.

## 3.1 β fe and β fd as weighted sums of ATEs of switching groups

We first show that under an additional condition, β fe and β fd can be written as a weighted sum of the ATEs of switching groups.

Assumption S2 (Stable treatment effect) For all g and t ≥ 2 ,

<!-- formula-not-decoded -->

The stable treatment effect assumption requires that the ATE of every group treated in t -1 does not change from t -1 to t . By iteration, the ATE of a group treated for instance from period t 0 to T is unrestricted before t 0 but should be constant from t 0 to T . Assumption S2 rules out the possibility that the treatment effect changes over time. Therefore, it may be implausible and should be carefully discussed.

/negationslash

<!-- formula-not-decoded -->

We now show that under the common trend and stable treatment effects assumptions, β fe and β fd may identify weighted averages of ATEs. Let N S = ∑ ( g,t ): D g,t = D g,t -1 N g,t and, for all g and t ≥ 2 ,

Theorem S2 Suppose that Assumptions 1-5 and S2 hold. Then,

/negationslash

<!-- formula-not-decoded -->

/negationslash

Moreover, w S fd,g,t ≥ 0 for all g and t ≥ 2 . If Assumption 6 holds and N g,t /N g,t -1 does not vary across g for all t ≥ 2 , w S g,t ≥ 0 for all g and t ≥ 2 .

Theorem S2 shows that in sharp designs, under the common trends and stable treatment effect assumptions, β fe and β fd identify weighted sums of ATEs of switching cells. The weights differ from those in Theorems 1 and 2. Now the weights attached to β fe are all positive in staggered adoption designs, while the weights attached to β fd are all positive in all sharp designs. Therefore, in staggered adoption (resp. sharp) designs, β fe (resp. β fd ) relies on the assumption that the treatment effect is stable over time, but it does not require that treatment effects be homogeneous between groups.

## 3.2 Non-binary, ordered treatment

We now consider the case where the treatment takes a finite number of ordered values, D i,g,t ∈ { 0 , 1 , ..., d } , and show that Theorem 1 can easily be extended to this case. 5 We need to define

5 Theorem 2 can also be extended to the case of a non-binary treatment.

potential outcomes for all the possible treatment values. For instance Y i,g,t ( d ) is the counterfactual outcome of observation i in cell ( g, t ) if she receives treatment value d . We also need to modify the treatment effect parameters we consider. In lieu of δ TR , we consider the average causal response (ACR) on the treated,

<!-- formula-not-decoded -->

Similarly, for all ( g, t ) such that D g,t = 0 , we consider, instead of ∆ g,t ,

/negationslash

<!-- formula-not-decoded -->

Then, similarly to (2), the following decomposition holds:

/negationslash

Let w O g,t = ε g,t ∑ g,t Ng,t D g,t N 1 ε g,t . Note that if the treatment is binary, w O g,t = w g,t .

<!-- formula-not-decoded -->

Theorem S3 Suppose that Assumptions 1-5 hold and D i,g,t ∈ { 0 , ..., d } . Then,

/negationslash

<!-- formula-not-decoded -->

Theorem S3 shows that under Assumption 5, when the treatment is not binary β fe identifies a weighted sum of the ACRs in all the ( g, t ) cells that are not untreated. Then, since the proof of Corollary 1 does not rely on the nature of the treatment, Corollary 1 directly applies to ordered treatments as well, by just replacing w g,t and N g,t by w O g,t and N g,t D g,t , respectively. Corollary 2 extends as well to this set-up, by simply modifying the no-correlation condition appropriately.

## 3.3 Including covariates

Often times, researchers also include a vector of covariates X g,t as control variables in their regression. In this section, we show that our Theorem 1 can be extended to this case. 6 We start by redefining Regression 1 in this context.

6 Theorem 2 can also be extended to regressions with covariates.

Regression 1X (Fixed-effects regression with covariates)

Let ̂ β X fe denote the coefficient of D g,t in an OLS regression of Y i,g,t on group and period fixed effects, D g,t , and X g,t . Let β X fe = E ( ̂ β X fe ) . Then, we need to modify Assumptions 3-5. Hereafter, we let X g = ( X g, 1 , ..., X g,t ) .

Assumption S3 (Independent groups with covariates) The vectors ( Y g,t (0) , Y g,t (1) , D g,t , X g,t ) 1 ≤ t ≤ T are mutually independent.

Assumption S4 (Strong exogeneity and common trends with covariates) There is a vector γ of same dimension as X g,t such that

<!-- formula-not-decoded -->

E ( Y g,t (0) -Y g,t -1 (0) -( X g,t -X g,t -1 ) γ ) g

Rearranging, Assumption S4 requires that

<!-- formula-not-decoded -->

for some constant λ t . Then, Assumption S4 allows for the possibility that groups experience different evolutions of their Y g,t (0) over time, but it requires that those differential evolutions are fully accounted for by a linear model in X g,t -X g,t -1 , the change in a group's covariates. Assumption S4 is implied by the linear model that is often invoked to justify the use of the FE regression with covariates. For instance, the use of Regression 1X is often justified by the following model:

<!-- formula-not-decoded -->

Equation (24) implies Assumption S4, but it does not imply Assumption 5.

An interesting special case is when the control variables are group-specific linear trends. Then, Assumption s4 requires that for all t ≥ 2 ,

<!-- formula-not-decoded -->

for some constants γ g and λ t . From t -1 to t , the evolution of Y (0) in group g should deviate from its group-specific linear trend γ g by an amount λ t common to all groups. Then, Assumption S4 is a 'common deviation from linear trends' assumption, which may be more plausible than the standard common trends assumption.

/negationslash

Let ε X g,t denote the residual of observations in cell ( g, t ) in the regression of D g,t on group and period fixed effects and X g,t . One can show that if the regressors in Regression 1 are not collinear, the average value of ε X g,t across all treated ( g, t ) cells differs from 0: ∑ ( g,t ): D g,t =1 ( N g,t /N 1 ) ε X g,t = 0 . Then, let

<!-- formula-not-decoded -->

Theorem S4 Suppose that Assumptions 1-2 and S3-S4 hold. Then,

<!-- formula-not-decoded -->

Theorem S4 shows that under a modified version of the common trends assumption accounting for the covariates, β X fe identifies a weighted sum of the ∆ TR g,t s, as β fe in Theorem 1, with different but still potentially negative weights. 7 Assumption S4 may be more plausible than Assumption 5, but adding covariates may increase the prevalence of negative weights, or the correlation between the weights and the ∆ g,t s, thus making β X fe less robust to heterogeneous effects than β fe .

## 3.4 2SLS regressions

Researchers have sometimes estimated 2SLS versions of Regressions 1 and 2. Our main conclusions also apply to those regressions. Let ̂ β 2 SLS fe denote the coefficient of D i,g,t in a 2SLS regression of Y g,t on group and period fixed effects and D i,g,t , using a variable Z g,t constant within each group × period as the instrument for D i,g,t . Z g,t typically represents an incentive for treatment allocated at the group × period level. For instance, Duflo (2001) studies the effect of years of schooling on wages in Indonesia, using a primary school construction program as an instrument. Specifically, she estimates a 2SLS regression of wages on cohort and district of birth fixed effects and years of schooling, using the interaction of belonging to a cohort entering primary school after the program was completed and the number of schools constructed in one's district of birth as the instrument for years of schooling.

Remark that ̂ β 2 SLS fe = ̂ β Y fe / ̂ β D fe , where ̂ β Y fe (resp. ̂ β D fe ) is the coefficient of Z g,t in the reducedform regression of Y g,t (resp. the first-stage regression of D g,t ) on group and period fixed effects and Z g,t . Then let β 2 SLS fe = E [ ̂ β Y fe ] /E [ ̂ β D fe ] . 8 Following Imbens and Angrist (1994), for any z ∈ Supp ( Z ) let D i,g,t ( z ) denote the potential treatment of unit i in ( g, t ) if Z i,g,t = z . It follows from Theorem 1 that under a common trends assumption on D i,g,t (0) , E [ ̂ β D fe ] is equal to a weighted sum of the average effects of the instrument on the treatment in each group and time period, with potentially many negative weights. Similarly, under a common trends assumption

7 In a previous version of this paper, we had shown that under a different, and arguably less natural, common trends assumption, β X fe identifies a weighted sum of the ∆ TR g,t , with the same weights as in Theorem 1. We thank an anonymous referee for pointing out issues with the common trends assumption we had previously proposed.

8 We do not consider here E [ ̂ β 2 SLS fe ] , as the 2SLS estimator may not have an expectation. Moreover, under conditions similar to those imposed in Section 5 of the paper, β 2 SLS fe is the probability limit of ̂ β 2 SLS fe , which makes β 2 SLS fe the proper estimand here.

on Y i,g,t ( D i,g,t (0)) instead of Y i,g,t (0) , E [ ̂ β Y fe ] is equal to a weighted sum of the average effects of the instrument on the outcome, again with potentially many negative weights. For instance, in Duflo (2001), under a common trends assumption on D i,g,t (0) , the number of years of schooling individuals would complete if zero new schools were constructed in their district, the first stage coefficient identifies a weighted sum of the effect of one new school on years of schooling in every district, with many negative weights. 9

Hence, it is only if the average effects of Z g,t on Y i,g,t and D i,g,t are constant across groups and periods, or if the weights are uncorrelated to treatment effects as in Assumption 7, that the reduced-form and first-stage coefficients respectively identify the average effect of Z i,g,t on Y i,g,t and D i,g,t . Then, this implies that β 2 SLS fe identifies, under the conditions in Imbens and Angrist (1994), the LATE of D i,g,t on Y i,g,t among units that comply with the instrument. 10

## 4 Extending the DID M estimator

Theorem 3 can be extended to the case where D i,g,t is not binary but takes values in D = { 0 , ..., d } . The causal effect we consider is the switchers' causal response

/negationslash where N D,S = ∑ ( g,t ): t ≥ 2 N g,t | D g,t -D g,t -1 | . Note that δ SCR = δ S when D i,g,t is binary.

<!-- formula-not-decoded -->

We identify δ SCR under the following two conditions, which generalize Assumptions 4-5 and 9-12 to non-binary treatments.

Assumption S5 (Mean independence between a group's outcome and other groups treatments)) For all ( d, g, t ) ∈ D × { 1 , ..., G } × { 1 , ..., T } , E ( Y g,t ( d ) | D ) = E ( Y g,t ( d ) | D g ) .

Assumption S6 (Strong exogeneity) For all ( d, g, t ) ∈ D × { 1 , ..., G } × { 2 , ..., T } , E ( Y g,t ( d ) -Y g,t -1 ( d ) | D g ) = E ( Y g,t ( d ) -Y g,t -1 ( d )) .

Assumption S7 (Common trends) For every d , for all t ≥ 2 and g , E ( Y g,t ( d ) -Y g,t -1 ( d )) does not vary across g .

9 New schools were constructed in every district, so this application falls into the heterogeneous adoption case.

10 In the special case with two groups and two periods, a binary incentive for treatment, and where only group 1 in period 1 receives the incentive, de Chaisemartin (2010) and Hudson et al. (2015) show that in a 2SLS regression of Y i,g,t on 1 { g = 2 } , 1 { t = 2 } and D i,g,t , using Z g,t = 1 { g = 2 } 1 { t = 2 } as the instrument, the coefficient of D i,g,t identifies a LATE under common trends assumptions on Y i,g,t ( D i,g,t (0)) and D i,g,t (0) . However, the discussion above shows that this result does not generalize to applications with multiple groups and periods or a non-binary instrument, as in Duflo (2001) where the number of new schools constructed varies across districts.

/negationslash

Assumption S8 (Existence of 'stable' groups) For all t ∈ { 2 , ..., T } , for all ( d, d ′ ) ∈ D 2 , d = d ′ , if there is at least one g ∈ { 1 , ..., G } such that D g,t -1 = d and D g,t = d ′ , then there exists at least one g ′ = g, g ′ ∈ { 1 , ..., G } such that D g ′ ,t -1 = D g ′ ,t = d .

When the treatment takes a large number of values, Assumption S8 may be violated. A solution, then, is to consider a modified treatment variable ˜ D g,t = h ( D g,t ) that groups together several values of D g,t , to ensure that Assumption S8 holds for ˜ D g,t . For instance, if the treatment can be equal to 0, 1, 2, or 3, and there is a group whose treatment switches from 2 to 3 between periods 1 and 2, but no group whose treatment remains equal to 2 between those two dates, one may define ˜ D g,t = min( D g,t , 2) if there is a group whose treatment is equal to 3 at periods 1 and 2. Then, Theorem S5 below still holds, after replacing D g,t by ˜ D g,t in the DID d,d ′ ,t estimators defined below, and if Assumption S7 is replaced by the requirement that E ( Y g,t ( d ) -Y g,t -1 ( d )) only depends on t and h ( d ) .

In order to define DID M in this context, let us introduce, for all ( d, d ′ , t ) ∈ D 2 ×{ 2 , ..., T } ,

<!-- formula-not-decoded -->

where N d,d ′ ,t is defined as in (3) for any ( d, d ′ ) ∈ D 2 . Then

/negationslash

<!-- formula-not-decoded -->

If the treatment is binary, the DID M estimator defined above is equal to that defined in Section 4 of the paper.

Theorem S5 Suppose that D i,g,t ∈ D and Assumptions 1-2 and S5-S8 hold. Then E [ DID M ] = δ SCR .

Theorem S5 generalizes Theorem 3 to non-binary treatments. We can also extend Theorem 4 in the same way to construct placebo tests of Assumption S7.

Finally, Theorem 3 can also be extended to the case with covariates. Under versions of Assumptions 10 and 11 written conditional on X , a conditional version of the DID M estimator is consistent for δ S under the common support condition Supp ( X d,g,t ) = Supp ( X ) . We refer to de Chaisemartin and D'Haultfœuille (2018) for further details.

/negationslash

## 5 Statistical properties of DID M and inference on δ S

In this section, we establish the asymptotic properties of DID M and construct confidence intervals on δ S based on DID M . We consider an asymptotic framework where the number of groups G tends to infinity. To define the confidence intervals, let P d,d ′ ,t = N d,d ′ ,t /G and

<!-- formula-not-decoded -->

/negationslash

Then, let σ 2 = ∑ g ̂ ψ 2 g /G , with

<!-- formula-not-decoded -->

We consider confidence intervals of the form where z 1 -α/ 2 denotes the quantile of order 1 -α/ 2 of a standard normal variable.

<!-- formula-not-decoded -->

We now establish the asymptotic properties of DID M and CI 1 -α ( δ S ) under the following assumptions. Hereafter, we denote U = ( P 0 , 0 , 1 , Q 0 , 0 , 1 , ..., P 1 , 1 ,T , Q 1 , 1 ,T ) .

Assumption S9 (Existence of moments and limits) sup g,t N g,t &lt; + ∞ and sup ( d,g,t ) E ( Y 4 g,t ( d )) &lt; + ∞ . lim G E [ U ] and lim G G × V ( U ) exist.

Assumption S10 (Positive probability of 'stable' groups and existence of switchers) For all ( d, g, t ) ∈ { 0 , 1 }× ( N \{ 0 } ) ×{ 2 , ..., T } , Pr( D g,t = 1 -d, D g,t -1 = d ) &gt; 0 implies lim G E [ P 1 -d,d,t ] &gt; 0 and lim G E [ P d,d,t ] &gt; 0 . Moreover, lim G E [ P 0 , 1 ,t + P 1 , 0 ,t ] &gt; 0 for at least one t .

Assumption S9 imposes the (uniform) existence of moments of order 4 of Y g,t ( d ) , and that some non-random averages converge as G tends to infinity. These assumptions ensure that we can apply law of large numbers and central limit theorems in our set-up where groups are independent but not necessarily identically distributed. Assumption S10 imposes that when at least one group switches from d to 1 -d with a positive probability on a given period, then on average over all groups, the limit probabilities of switching from d to 1 -d will be positive as G →∞ . The limit probability of remaining at d will also be positive. This latter condition may be seen as a weaker version of Assumption 11, as it imposes the existence of 'stable' groups only

with probability tending to one as G →∞ . The last condition in Assumption S10 simply states that asymptotically, the proportion of switchers is strictly postive.

The following result shows that under these conditions, DID M is asymptotically normal, and that CI 1 -α ( δ S ) is asymptotically conservative.

Theorem S6 Suppose that Assumptions 1-5, 9-10 and S9-S10 hold. Then, as G →∞ , with σ 2 defined in (38) below. Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem S6 shows that DID M is an asymptotically normal estimator of δ S when the number of groups tends to infinity, provided the outcomes and treatments are independent across groups. As is usually the case for estimators constructed using independent but not identically distributed random variables (see e.g. Liu and Singh, 1995), the asymptotic variance of DID M can only be conservatively estimated. As a result, the confidence interval we propose is asymptotically conservative.

## 6 Detailed literature review

We now review the 33 papers that use two-way fixed effects or closely related regressions that we found in our literature review. For each paper, we use the following presentation:

Authors (year), Title. Where the two-way fixed effects estimator is used in the paper.

Description of the two-way fixed effects estimator used in the paper, and how it relates to Regression 1 or 2. Assessment of whether the stable groups assumption holds in this paper. Assessment of whether the research design is sharp or fuzzy.

1. Chandra et al. (2010), Patient Cost-Sharing and Hospitalization Offsets in the Elderly. First line of Tables 2 and 3.

In the regressions in the first line of Tables 2 and 3, the outcomes (e.g. a measure of utilization for plan p in month t) are regressed on plan fixed effects, month fixed effects, and an indicator of whether plan p had increased copayments in month t (see regression equation at the bottom of page 198). This regression corresponds to Regression 1. The period analyzed runs from January 2000 to September 2003. The stable groups assumption is satisfied until January 2002, when the HMO plans also become treated. This is a sharp design.

2. Duggan and Morton (2010), The Effect of Medicare Part D on Pharmaceutical Prices and Utilization. Tables 2 and 3.

In regression Equation (1), the dependent variable is the change in the price of drug j between 2003 and 2006, the explanatory variables are the Medicare market share for drug j in 2003, and some control variables. This regression corresponds to Regression 2, with some control variables. The stable groups assumption is presumably not satisfied: it seems unlikely that there are drugs whose Medicare market share in 2003 is equal to 0. This is a sharp design.

3. Aizer (2010), The Gender Wage Gap and Domestic Violence. Table 2.

In regression Equation (2), the dependent variable is the log of female assaults among females of race r in county c and year t, and the explanatory variables are race, year, county, race × year, race × county, and county × year fixed effects, as well as the gender wage gap in county c, year t, and race r, and some control variables. This regression is a 'three-way fixed effects' version of Regression 1, with some control variables. The stable groups assumption is presumably satisfied: it seems likely that between each pair of consecutive years, there are counties where the gender wage gap does not change. This is a fuzzy design: the treatment of interest is the gender wage gap in a couple (see the bargaining model in Appendix 1), which varies within (year,county) cells.

4. Algan and Cahuc (2010), Inherited Trust and Growth. Figure 4.

Figure 4 presents a regression of changes in income per capita from 1935 to 2000 on changes in inherited trust over the same period and a constant. This regression corresponds to Regression 2. The stable groups assumption is satisfied: there are countries where inherited trust does not change from 1935 to 2000. This is a sharp design.

5. Ellul et al. (2010), Inheritance Law and Investment in Family Firms. Table 7. In the regressions presented in Table 7, the dependent variable is the capital expenditure of firm j in year t, and the explanatory variables are firm fixed effects, an indicator for whether year t is a succession period for firm j, some controls, and three treatment variables: the interaction of the succession indicator with the level of investor protection in the country where firm j is located, the interaction of the succession indicator with the level of inheritance laws permissiveness in the country where firm j is located, and the interaction of the succession indicator with the level of inheritance laws permissiveness and the level of investor protection in the country where firm j is located. This regression is similar to Regression 1 with controls, except that it has three treatment variables. The stable groups assumption is presumably not satisfied: for instance, it seems unlikely that there are countries with no investor protection at all. This is a sharp design.
6. Bustos (2011), Trade Liberalization, Exports, and Technology Upgrading: Ev-

## idence on the Impact of MERCOSUR on Argentinean Firms. Tables 3 to 12.

In regression Equation (11), the dependent variable is the change in exporting status of firm i in sector j between 1992 and 1996, and the explanatory variables are the change in trade tariffs in Brasil for products in sector j over the same period, and some control variables. This regression corresponds to Regression 2, with some controls. The stable groups assumption is presumably satisfied: it seems likely that there are sectors where trade tariffs in Brasil did not change between 1992 and 1996. This is a sharp design.

7. Anderson and Sallee (2011), Using Loopholes to Reveal the Marginal Cost of Regulation: The Case of Fuel-Economy Standards. Table 5 Column 2.

In the regression in Table 5 Column (2), the dependent variable is an indicator for whether a car sold is a flexible fuel vehicle, and the explanatory variables are state and month fixed effects, the percent of gas stations that have ethanol fuel in each month × state, and some controls. This regression corresponds to Regression 1. The stable groups assumption is presumably satisfied: it seems likely that between each pair of consecutive months, there are states where the percent ethanol availability does not change. This a fuzzy design: the treatment of interest is whether a car buyer has access to ethanol fuel, which varies within (month,state) cells.

8. Bagwell and Staiger (2011), What Do Trade Negotiators Negotiate About?

Empirical Evidence from the World Trade Organization. Table 3, OLS columns. In regression equations (15a) and (15b), the dependent variable is the ad valorem tariff level bound by country c on product g, while the explanatory variables are country and product fixed effects, and two treatment variables which vary at the country × product level. These regressions are similar to Regression 1, except that they have two treatment variables. The stable groups assumption is not applicable here, as none of the two sets of fixed effects included in the regression correspond to an ordered variable. This is a sharp design.

9. Zhang and Zhu (2011), Group Size and Incentives to Contribute: A Natural Experiment at Chinese Wikipedia. Tables 3 and 4, Columns 4-6.

In the regression in, say, Table 3 Column (4), the dependent variable is the total number of contributions to Wikipedia by individual i at period t, regressed on individual fixed effects, an indicator for whether period t is after the Wikipedia block, the interaction of this indicator and a measure of social participation by individual i, and some controls. This regression corresponds to Regression 1 with some controls. The stable groups assumption is satisfied: there are individuals with a social participation measure equal to 0. This is a sharp design.

10. Hotz and Xiao (2011), The Impact of Regulations on the Supply and Quality

of Care in Child Care Markets. Table 7, Columns 4 and 5.

In Regression Equation (1), the dependent variable is the outcome for market m in state s and year t, and the explanatory variables are state and year fixed effects, various measures of regulations in state s in year t, and some controls. This regression corresponds to Regression 1 with several treatment variables and with some controls. The stable groups assumption is presumably satisfied: between each pair of consecutive years, it is likely that there are states whose regulations do not change. This is a sharp design.

11. Mian and Sufi (2011), House Prices, Home Equity-Based Borrowing, and the US Household Leverage Crisis. Tables 2 and 3.

In Regression Equation (1), the dependent variable is the change in homeowner leverage from 2002 to 2006 for individual i living in zip code z in MSA m, and the dependent variable is the change in the house price for that individual, instrumented by MSA-level housing supply elasticity. This regression is the 2SLS version of Regression 2, with some controls. The stable groups assumption is presumably not satisfied: it is unlikely that some MSAs have an housing supply elasticity equal to 0. This is a sharp design.

12. Wang (2011), State Misallocation and Housing Prices: Theory and Evidence from China. Table 5, Panel A.

In regression Equation (15), the dependent variable is the quantity of housing services in household i's residence in year t, while the explanatory variables are an indicator for period t being after the reform, a measure of mismatch in household i, the interaction of the measure of mismatch and the time indicator, and some controls. This regression is similar to Regression 1 with some controls, except that it has a measure of mismatch in household i instead of household fixed effects. The stable groups assumption is presumably satisfied: it is likely that some households have a mismatch equal to 0. This is a sharp design.

13. Duranton and Turner (2011), The Fundamental Law of Road Congestion: Evidence from US Cities. Table 5.

In the regressions presented in, say, the first column of Table 5, the dependent variable is the change in vehicle kilometers traveled in MSA s between decades t and t-1, and the explanatory variables are the change in kilometers of roads in MSA s between decades t and t-1, and decade effects. This regression corresponds to Regression 2. The stable groups assumption is presumably satisfied: it is likely that between each pair of consecutive decades, there are some MSAs where the kilometers of roads do not change. This is a sharp design.

14. Acemoglu et al. (2011), The Consequences of Radical Reform: The French Revolution. Table 3.

In regression Equation (1), the dependent variable is urbanization in polity j at time t, while the explanatory variables are time and polity fixed effects, and the number of years of French presence in polity j interacted with the time effects. This regression corresponds to Regression 1. The stable groups assumption is satisfied as there are several polities that did not experience any year of French presence. This is a sharp design.

15. Baum-Snow and Lutz (2011), School Desegregation, School Choice, and Changes in Residential Location Patterns by Race. Tables 2 to 6.

In regression Equation (1), the dependent variable is, say, whites public school enrolment in MSA j in year t, while the explanatory variables are MSA and region × time fixed effects, and an indicator for whether MSA j is desegregated. This regression corresponds to Regression 1 with controls. The stable groups assumption is satisfied: between each pair of consecutive years, there are MSAs whose desegregation status does not change. This is a sharp design.

16. Dinkelman (2011), The Effects of Rural Electrification on Employment: New Evidence from South Africa. Tables 4 and 5 Columns 5-8, Table 8 Columns 3-4, Table 9 Column 2, and Table 10 Columns 2, 4, and 6.

In regression Equation (3), the dependent variable is, say, the first difference of the female employment rate for community j between periods 1 and 2, and the explanatory variables are district fixed effects, the change of electrification status of community j between periods 1 and 2, and some statistical controls. The land gradient in community j is used as an instrument for the change in electrification. This regression corresponds to the 2SLS version of Regression 2 with some controls. The stable groups assumption is presumably satisfied: it is likely that there are communities whose land gradient is 0. This is a sharp design.

17. Enikolopov et al. (2011), Media and Political Persuasion: Evidence from Russia. Table 3.

In regression Equation (5), the dependent variable is the share of votes for party j in election-year t and subregion s, and the explanatory variables are subregion and election fixed effects, and the share of people having access to NTV in subregion s in election-year t. This regression corresponds to Regression 1. The stable groups assumption is not satisfied: the share of people having access to NTV strictly increases in all regions between 1995 and 1999, the two elections used in the analysis. This a fuzzy design: the treatment of interest is whether a person has access to NTV, which varies within (subregion,year) cells.

18. Fang and Gavazza (2011), Dynamic Inefficiencies in an Employment-Based Health Insurance System: Theory and Evidence. Tables 2, 3, 5, and 6, Column 3. In regression Equation (7), the dependent variable is the health expenditures of individual j working in industry i in period t and region r, and the explanatory variables are individual

effects, region specific time effects, and the job tenure of individual j. The death rate of establishments in industry i in period t and region r is used as an instrument for the job tenure of individual j. This regression is the 2SLS version of Regression 2 with controls. The stable groups assumption is presumably satisfied: between each pair of consecutive years, it is likely that there are some industry × region pairs where the death rate of establishments does not change. This a fuzzy design: the instrument of interest is whether a person's former employee closed down over the current year, which varies within (industry,year) cells.

## 19. Gentzkow et al. (2011), The Effect of Newspaper Entry and Exit on Electoral Politics. Tables 2 and 3.

In regression Equation (2), the dependent variable is the change in voter turnout in county c between elections year t and t-1, and the explanatory variables are state × year effects, and the change in the number of newspapers in county c between t and t-1. This regression corresponds to Regression 2 with controls. The stable groups assumption is satisfied: between each pair of consecutive years, there are some counties where the number of newspapers does not change. This is a sharp design.

20. Bloom et al. (2012), Americans Do IT Better: US Multinationals and the Productivity Miracle. Table 2, Columns 6-8.

In the regression in, say, Column 6 of Table 2, the dependent variable is the log of output per worker in firm i in period t, while the explanatory variables are firms and time fixed effects, the log of the amount of IT capital per employee ln( C/L ) , the interaction of ln( C/L ) and an indicator for whether the firm is owned by a US multinational, the interaction of ln( C/L ) and an indicator for whether the firm is owned by a non-US multinational, and some controls. This regression is similar to Regression 1 with some controls, except that it has three treatment variables. The stable groups assumption is presumably satisfied: between each pair of consecutive years, it is likely that there are some firms where the amount of IT capital per employee ln( C/L ) does not change. This is a sharp design.

## 21. Simcoe (2012), Standard Setting Committees: Consensus Governance for Shared Technology Platforms. Table 4, Columns 1-3.

In regression Equation (5), the dependent variable is a measure of time to consensus for project i submitted to committee j, while the explanatory variables are an indicator for projects submitted to the standards track, a measure of distributional conflict, the interaction of the standards track and distributional conflict, and some controls variables. This regression is similar to Regression 1 with some controls, except that it has a measure of distributional conflict instead of committee fixed effects. The stable groups assumption is presumably not satisfied: it is unlikely that there is any committee where the measure of

distributional conflict is equal to 0. This is a sharp design.

22. Moser and Voena (2012), Compulsory Licensing: Evidence from the Trading with the Enemy Act. Table 2.

In the regression equation in the beginning of Section III, the dependent variable is the number of patents by US inventors in patent class c at period t, and the explanatory variables are patent class and time fixed effects, the interaction of period t being after the trading with the enemy act and the number of licensed patents in class c, and some control variables. This regression corresponds to Regression 1 with some controls. The stable groups assumption is satisfied: there are patent classes where no patent was licensed. This is a sharp design.

23. Forman et al. (2012), The Internet and Local Wages: A Puzzle. Tables 2 and 4. In regression Equation (1), the dependent variable is the difference between log wages in 2000 and 1995 in county i, and the explanatory variables are the proportion of businesses using Internet in county i in 2000, and control variables. This regression corresponds to Regression 2 with some controls. The stable groups assumption is satisfied: there are counties with no Internet investment in 2000. This a fuzzy design: the treatment of interest is whether a business uses Internet, which varies within (county,year) cells.

24. Besley and Mueller (2012), Estimating the Peace Dividend: The Impact of Violence on House Prices in Northern Ireland. Table 1, Columns 3 and 5-7. In regression Equation (1), the dependent variable is the price of houses in region r at time t, while the explanatory variables are region and time fixed effects, and the number of people killed because of the civil war in region r at time t-1. This regression corresponds to Regression 1. The stable groups assumption is presumably satisfied: between each pair of consecutive years, it is likely that there are some regions where the number of people killed because of the civil war does not change. This is a sharp design.

25. Dafny et al. (2012), Paying a Premium on Your Premium? Consolidation in the US Health Insurance Industry. Table 3.

In regression Equation (3), the dependent variable is the the concentration of the hospital industry in market m and year t, and explanatory variables are time fixed effects, market fixed effects, and the change in concentration in market m induced by a merger interacted with an indicator for t being after the merger. This regression corresponds to Regression 1. The stable groups assumption is satisfied: there are many markets where the merger did not change concentration. This is a sharp design.

26. Hornbeck (2012), The Enduring Impact of the American Dust Bowl: Shortand Long-Run Adjustments to Environmental Catastrophe. Table 2. In regression

Equation (1), the dependent variable is, say, the change in log land value in county c between period t and 1930, and the explanatory variables are state × year fixed effects, the share of county c in high erosion regions, the share of county c in medium erosion regions, and some control variables. This regression is similar to Regression 1 with controls, except that it has two treatment variables. The stable groups assumption is satisfied: many counties have 0% of their land situated in medium or high erosion regions. This a fuzzy design: the treatments of interest are whether a piece of land is in high or in medium erosion regions, which varies within (county,year) cells.

27. Bajari et al. (2012), A Rational Expectations Approach to Hedonic Price Regressions with Time-Varying Unobserved Product Attributes: The Price of Pollution. Table 5.

In, say, the first regression equation in the bottom of page 1915, the dependent variable is the change in the price of house j between sales 2 and 3, and the explanatory variables are the change in various pollutants in the area around house j between sales 2 and 3, and some controls. This regression is similar to Regression 2 with controls, except that it has several treatment variables. The stable groups assumption is presumably satisfied: it is likely that for each pair of consecutive sales, there are houses where the level of each pollutant does not change. This is a sharp design.

28. Dahl and Lochner (2012), The Impact of Family Income on Child Achievement: Evidence from the Earned Income Tax Credit. Table 3.

In regression Equation (4), the dependent variable is the change in test scores for child i between years a and a-1, while the explanatory variables are the change in the EITC income of her family and some controls, and the change in the expected EITC income of her family based on her family income in year a-1 is used to instrument for the actual change of her family's EITC income. This regression is a 2SLS version of Regression 2 with controls, except that it does not have years fixed effects. The stable groups assumption is presumably satisfied: it is likely that for each pair of consecutive years, there are children whose family's expected EITC income does not change. This is a sharp design.

29. Imberman et al. (2012), Katrina's Children: Evidence on the Structure of Peer Effects from Hurricane Evacuees. Tables 3-6.

In regression Equation (1), the dependent variable is the test score of student i in school j in grade g and year t, and the explanatory variables are school and grade × year fixed effects, the fraction of Katrina evacuee students received by school j in grade g and year t, and some controls. This regression is a three-way fixed effects version of Regression 1. The stable groups assumption is satisfied: there are schools that did not receive any Katrina evacuee. This a fuzzy design: the treatment of interest is the proportion of evacuees in

one's class, which varies within (school,grade,year) cells.

## 30. Chaney et al. (2012), The Collateral Channel: How Real Estate Shocks Affect Corporate Investment. Table 5.

In regression Equation (1), the dependent variable is the value of investment in firm i and year t divided by the lagged book value of properties, plants, and equipments (PPE), and the explanatory variables are firm and time fixed effects and the market value of firm i in year t divided by its lagged PPE, and some controls. This regression corresponds to Regression 1, with some controls. The stable groups assumption is presumably satisfied: it is likely that between each pair of consecutive years, there are firms whose market value divided by their lagged PPE does not change. This is a sharp design.

31. Aaronson et al. (2012), The Spending and Debt Response to Minimum Wage Hikes. Tables 1, 2, and 5.

In regression Equation (1), the outcome variable is, say, income of household i at period t, and the explanatory variables are household and time fixed effects, and the minimum wage in the state where household i lives in period t. This regression corresponds to Regression 1. The stable groups assumption is satisfied: between each pair of consecutive periods, there are states where the minimum wage does not change. This is a sharp design.

## 32. Brambilla et al. (2012), Exports, Export Destinations, and Skills. Table 5.

In the regression in, say, the first column of Table 2, the dependent variable is a measure of skills in the labor force employed by firm i in industry j at period t, and the explanatory variables are firm and industry × period fixed effects, the ratio of exports to sales in firm i at period t, and some controls. This regression corresponds to Regression 1, with some controls. The stable groups assumption is presumably satisfied: it is likely that between each pair of consecutive periods, there are firms whose ratio of exports to sales does not change. This is a sharp design.

33. Faye and Niehaus (2012), Political Aid Cycles. Table 3, Columns 4 and 5, and Tables 4 and 5.

In regression Equation (2), the dependent variable is the amount of donations received by receiver r from donor d in year t, and the explanatory variables are donor × receiver fixed effects, an indicator for whether there is an election in country r in year t, a measure of alignment between the ruling political parties in countries r and d at t , and the interaction of the election indicator and the measure of alignment. This regression corresponds to Regression 1. The stable groups assumption is presumably not satisfied: it is unlikely that there are donor-receiver pairs that are perfectly unaligned. This is a sharp design.

## 7 Proofs

Theorem S1 relies on the following lemma.

Lemma S1 If Assumptions 1 and 3-5 hold, for all ( g, g ′ , t, t ′ ) ∈ { 1 , ..., G } 2 ×{ 1 , ..., T } 2 ,

<!-- formula-not-decoded -->

Proof of Lemma S1

For all ( g, t ) ∈ { 1 , ..., G } × { 1 , ..., T } ,

The end of the proof is the same as that of Lemma 1.

<!-- formula-not-decoded -->

## Proof of Theorem S1

The proof of that result is very similar to the proof of Theorem 1.

<!-- formula-not-decoded -->

The second equality follows from the law of iterated expectations. The third and fifth equalities follow from Equations (5) and (6). The fourth equality follows from Lemma S1. The last equality follows from the law of iterated expectations.

## Proof of Proposition S1

Assuming that N g, 2 /N g, 1 does not vary across g ensures that there exists a strictly positive real number φ such that N g, 2 /N g, 1 = φ . Then,

<!-- formula-not-decoded -->

where the first and third equalities follow from the fact N g, 2 /N g, 1 does not vary across g .

Then, the definition of w TR g, 2 , Equation (25) and Assumption 1 imply that

## Proof of Theorem S2

<!-- formula-not-decoded -->

The proof relies on the lemma below, which we start by proving, before proving the theorem.

Lemma S2 If Assumptions 1-5 and S2 hold,

<!-- formula-not-decoded -->

Proof of Lemma S2

By Lemma 1 and Assumption S2,

<!-- formula-not-decoded -->

Proof of the decomposition for the fixed-effect regression

First, we have

The first equality follows by (6). The second equality follows from summation by part and (6). The third equality follows from Lemma S2. The fourth equality stems from the fact that by (6), the terms with g = 1 vanish.

<!-- formula-not-decoded -->

Similarly,

The result follows by combining (4), (26), (27), and the law of iterated expectations.

/negationslash

<!-- formula-not-decoded -->

/negationslash

Proof of the decomposition for the first-difference regression

First, we have

The first equality follows from (14). The second equality follows from Lemma S2. The third equality follows from (14) again. The result follows by combining (13) with the last display, and using the law of iterated expectations.

<!-- formula-not-decoded -->

Proof that w S g,t ≥ 0 under Assumption 6 and if N g,t /N g,t -1 does not depend on g

Under Assumption 6, one has that D g,t = 1 { t ≥ a g } , with a g ∈ { 1 , ..., T +1 } . Therefore, given the form of w S g,t , we just have to prove that for all g ,

Because N g,t /N g,t -1 does not vary across g for all t ≥ 2 , we have N g,t = N g, 0 γ t for some γ t ≥ 0 . Moreover, ε g,t = D g,t -D g,. -D .,t + D .,. . Let ˜ γ t = γ t / ∑ t ≥ 0 γ t , then D g,. = ∑ t ≥ a g ˜ γ t , and D .,. = ∑ t ≥ 0 γ t D .,t . Hence,

Now, because D .,t ≤ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, in view of (29),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Therefore, (28) and the result follows.

Proof that w S fd,g,t ≥ 0

We just have to focus on the cases where D g,t = D g,t -1 . Note that ε fd,g,t = D g,t -D g,t -1 -( D .,t -D .,t -1 ) . Then, if D g,t -D g,t -1 = 1 , the numerator of w S fd,g,t has the same sign as 1 -( D .,t -D .,t -1 ) , which is positive. If D g,t -D g,t -1 = -1 , the numerator of w S fd,g,t has the same sign as 1 + ( D .,t -D .,t -1 ) , which is also positive. Because the denominator sums terms that are always positive, it is positive as well. The result follows.

## Proof of Theorem S3

The reasoning is exactly the same as in Theorem 1, except that we rely on Lemma S3 below, instead of Lemma 1. We thus only prove Lemma S3.

Lemma S3 If Assumptions 1-5 hold and D i,g,t ∈ { 0 , , ..., d } .

<!-- formula-not-decoded -->

Proof of Lemma S3

Under Assumption 2, we have E ( Y g,t | D ) = E ( Y g,t (0) | D )+ E ( Y g,t ( D g,t ) -Y g,t (0) | D ) . The result follows by decomposing similarly the three other terms E ( Y g,t ′ | D ) , E ( Y g ′ ,t | D ) , and E ( Y g ′ ,t ′ | D ) , using Assumptions 3-5, and finally using the definition of ∆ ACR g,t .

## Proof of Theorem S4

The proof relies on the following lemma, that resembles Lemma 1 and that we do not prove.

Lemma S4 If Assumptions 1, 2, and S3-S4 hold, for all ( g, g ′ , t, t ′ ) ∈ { 1 , ..., G } 2 ×{ 1 , ..., T } 2 ,

<!-- formula-not-decoded -->

It follows from the Frisch-Waugh theorem and the definition of ε X g,t that

<!-- formula-not-decoded -->

Now, by definition of ε X g,t again,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then,

The first equality follows from (33). The second follows from Equations (31) and (32). Hence,

<!-- formula-not-decoded -->

The first equality follows from Lemma 4. The second follows from Equations (31) and (32). The third follows from Assumption 2. Finally, Assumption 2 implies that

<!-- formula-not-decoded -->

Combining (30), (34), (35), and the law of iterated expectations yields the result.

## Proof of Theorem S5

Reasoning as in the proof of Theorem 3, we get that for all t ≥ 2 ,

<!-- formula-not-decoded -->

For all ( g, t ) , there exists one ( d, d ′ ) ∈ D 2 such that D g,t = d ′ and D g,t -1 = d . Hence,

/negationslash

<!-- formula-not-decoded -->

The result follows by definition of DID M and δ SCR , and the law of iterated expectations.

## Proof of Theorem S6

## 1. Asymptotic normality

Let us define P S = N S /G and

<!-- formula-not-decoded -->

/negationslash

We prove the result in two steps. First, we prove that √ G ( DID M -E ( T S ) /E ( P S ) ) is asymptotically normal. Second, we show that the difference between E ( T S ) /E ( P S ) and δ S is asymptotically negligible.

Convergence of √ G ( DID M -E ( T S ) /E ( P S ) ) By Assumption S9,

<!-- formula-not-decoded -->

Thus, Lyapunov's condition for the central limit theorem holds, and because Σ = lim g G × V ( U ) exists,

By Assumption S9, P ∞ d,d ′ ,t = lim G →∞ E ( P d,d ′ ,t ) exists. Then for d ∈ { 0 , 1 } , define T d = { t : P ∞ 1 -d,d,t &gt; 0 } and the event D = { N S &gt; 0 } ∩ D 0 ∩ D 1 , with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the law of large numbers and Assumption S10, N S &gt; 0 with probability approaching one. Next, fix d ∈ { 0 , 1 } . If t ∈ T d , then, by the law of large numbers, N 1 -d,d,t &gt; 0 with probability approaching one. Moreover, for such a t , there exists g such that Pr( D g,t = 1 -d, D g,t -1 = d ) &gt; 0 and thus, by Assumption S10, P ∞ d,d,t &gt; 0 . Then, by the law of large numbers again, N d,d,t &gt; 0 with probability approaching one. Conversely, if min( N d,d,t , N 1 -d,d,t ) &gt; 0 , then there exists g such that Pr( D g,t = 1 -d, D g,t -1 = d ) &gt; 0 . Hence, by Assumption S10 again, P ∞ 1 -d,d,t &gt; 0 . This shows that D d , and thus D , holds with probability approaching one.

<!-- formula-not-decoded -->

Now, by definition of DID M , under D we have DID M = f ( U ) , with for all ( p d,d ′ ,t ) ( d,d ′ ,t ) such that all denominators are strictly positive. By Assumption S9 again, E [ U ] converges to U ∞ . Furthermore, f is continuously differentiable in a neighborhood of U ∞ . Thus, by the uniform delta method (see, e.g. van der Vaart, 2000, Theorem 3.8),

<!-- formula-not-decoded -->

Finally, f ( E ( U )) = f N /f D , with

<!-- formula-not-decoded -->

Reasoning as in the proof of Theorem 3 (see in particular Equations (20)-(21)) and noting that if t /negationslash∈ T d , then, by Assumption S10 Pr( D g,t = 1 -d, D g,t -1 = d ) = 0 , we get f N = E ( T S ) . Moreover, f D = E ( P S ) . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Convergence to 0 of √ G ( E ( T S ) /E ( P S ) -δ S ) Let us define D G = ( D g,t ) ( g,t ): g ≤ G,t =1 ...T , ˜ T = E [ T S | D G ] and I = 1 {| P S -E ( P S ) | &lt; ε G } , where ε G &gt; 0 will be specified below. By Assumption S10, lim G →∞ E ( P S ) &gt; 0 . Thus, it suffices to prove that √ G ( E ( P S ) δ S -E ( T S ) ) → 0 . Because δ S = E [ T S /P S ] , we have

First, consider the second term on the right-hand side. By applying twice the Cauchy-Schwarz inequality, we get

<!-- formula-not-decoded -->

By Assumption S9, √ GV ( P S ) 1 / 2 converges towards a finite limit. Thus, it suffices to show that the term into brackets tends to 0. To this end, note first that

/negationslash

Now, let A g,t = N g,t 1 { D g,t = D g,t -1 } and B g,t = E [ Y g,t (1) -Y g,t (0) | D G ] . By Assumption S9 and Jensen's inequality, sup ( g,G,t ): g ≤ G E [ | B g,t | 4 ] &lt; + ∞ . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Hence, E [( ˜ T/P S ) 4 ] Pr( I = 0) ≤ K 1 G Pr( I = 0) for some constant K 1 &gt; 0 . Moreover, by Hoeffding's inequality,

By Assumption S9, there exists c &gt; 0 such that for all G , 1 /G ∑ G g =1 N 2 g,. &lt; 2 c . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ε G = ( C ln( G ) /G ) 1 / 2 , for some C &gt; c . Then, by what precedes, G Pr( I = 0) → 0 and the second term of the right-hand side of (39) tends to zero.

Now, let us move to the first term of the right-hand side of (39). We have

<!-- formula-not-decoded -->

We now prove that both terms on the right-hand side tend to zero. First, by Taylor expansions of x ↦→ 1 /x around E ( P S ) , there exist ( P S 1 , P S 2 ) in the interval between P S and E ( P S ) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When I = 1 , | P S 1 -E ( P S ) | &lt; ε G and | P S 2 -E ( P S ) | &lt; ε G . Recall also that lim G E ( P S ) &gt; 0 and ε G → 0 . Then, in view of (41) and by Assumption S9,

<!-- formula-not-decoded -->

Now, multiplying (42) by I and taking the expectation on both sides, we obtain:

Moreover, by definition of ˜ T , GV ( ˜ T ) ≤ GV ( T S ) , and the latter is bounded by Assumption S9. Therefore, the first term of the right-hand side of (40) tends to 0.

<!-- formula-not-decoded -->

## 2. Validity of the confidence intervals

There exists σ 2 G such that ∑ g ̂ ψ 2 g /G -σ 2 G P -→ 0 , with lim inf G σ 2 G ≥ σ 2 . Let Q ∞ d,d ′ ,t = lim G →∞ E [ Q d,d ′ ,t ] and λ g be the column vector such that U = ∑ G g =1 λ g /G . Some tedious algebra show that J f ( U ∞ ) × λ g = ψ g , with

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Hence, in view of (37)-(38), σ 2 = lim G ∑ g V ( ψ g ) /G . Next, we show that ∑ g ̂ ψ 2 g /G is asymptotically larger than σ 2 . For that purpose, let σ 2 G = ∑ g E ( ψ 2 g ) /G and remark that ̂ ψ g = J f ( U ) × λ g . Then

Let λ k,g denote the k th coordinate of λ g . Assumption S9 ensures that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we also have sup g E ( ψ 2 g ) &lt; + ∞ . Therefore, by the weak law of large numbers, the second term on the right-hand side of (43) converges to 0. Next,

By (44) again and the weak law of large numbers, U P -→ U ∞ . Moreover, f is continuously differentiable in a vicinity of U ∞ , Thus, by the continuous mapping theorem, J f ( U ) -J f ( U ∞ ) P -→ 0 . By (44) once again and the weak law of large numbers,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, by the Cauchy-Schwarz inequality and (44), we have, for all k, /lscript ,

∣ ∣ As a result, ∑ g λ g λ ′ g /G = O P (1) . Finally, because J f ( U ) converges in probability, J f ( U ) + J f ( U ∞ ) = O P (1) . Thus, in view of (45), the first term on the right-hand side of (43) converges in probability to 0. Hence, we have proven that ∑ g ̂ ψ 2 g /G -σ 2 G P -→ 0 . Finally, E ( ψ 2 g ) ≥ V ( ψ g ) and thus σ 2 G -∑ g V ( ψ g ) /G ≥ 0 . Therefore, lim inf G σ 2 G ≥ σ 2 .

CI 1 -α ( δ S ) is asymptotically conservative.

<!-- formula-not-decoded -->

By (38), the convergence to 0 of √ G ( E ( T S ) /E ( P S ) -δ S ) , (37), and since ψ g = J f ( U ∞ ) × λ g ,

Let Z G denote the term into brackets. By the first step above and Slutsky's lemma Z G d -→ N (0 , 1) . Fix η &gt; 0 . Because lim inf G σ 2 G ≥ σ 2 , there exists G 0 such that for every G ≥ G 0 ,

Then, letting Φ denote the cdf of the standard normal distribution, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The result follows by letting η tend to zero.

## References

- Aaronson, D., Agarwal, S. and French, E. (2012), 'The spending and debt response to minimum wage hikes', American Economic Review 102 (7), 3111-3139.
- Acemoglu, D., Cantoni, D., Johnson, S. and Robinson, J. A. (2011), 'The consequences of radical reform: The french revolution', American Economic Review 101 (7), 3286-3307.
- Aizer, A. (2010), 'The gender wage gap and domestic violence', American Economic Review 100 (4), 1847-1859.
- Algan, Y. and Cahuc, P. (2010), 'Inherited trust and growth', American Economic Review 100 (5), 2060-2092.
- Anderson, S. T. and Sallee, J. M. (2011), 'Using loopholes to reveal the marginal cost of regulation: The case of fuel-economy standards', American Economic Review 101 (4), 1375-1409.
- Bagwell, K. and Staiger, R. W. (2011), 'What do trade negotiators negotiate about? empirical evidence from the world trade organization', American Economic Review 101 (4), 1238-1273.
- Bajari, P., Fruehwirth, J. C., Kim, K. I. and Timmins, C. (2012), 'A rational expectations approach to hedonic price regressions with time-varying unobserved product attributes: The price of pollution', American Economic Review 102 (5), 1898-1926.
- Baum-Snow, N. and Lutz, B. F. (2011), 'School desegregation, school choice, and changes in residential location patterns by race', American Economic Review 101 (7), 3019-3046.
- Besley, T. and Mueller, H. (2012), 'Estimating the peace dividend: The impact of violence on house prices in northern ireland', American Economic Review 102 (2), 810-833.
- Bloom, N., Sadun, R. and Van Reenen, J. (2012), 'Americans do it better: Us multinationals and the productivity miracle', American Economic Review 102 (1), 167-201.
- Brambilla, I., Lederman, D. and Porto, G. (2012), 'Exports, export destinations, and skills', American Economic Review 102 (7), 3406-3438.
- Bustos, P. (2011), 'Trade liberalization, exports, and technology upgrading: Evidence on the impact of mercosur on argentinian firms', American Economic Review 101 (1), 304-340.
- Chandra, A., Gruber, J. and McKnight, R. (2010), 'Patient cost-sharing and hospitalization offsets in the elderly', American Economic Review 100 (1), 193-213.
- Chaney, T., Sraer, D. and Thesmar, D. (2012), 'The collateral channel: How real estate shocks affect corporate investment', American Economic Review 102 (6), 2381-2409.

- Dafny, L., Duggan, M. and Ramanarayanan, S. (2012), 'Paying a premium on your premium? consolidation in the us health insurance industry', American Economic Review 102 (2), 11611185.
- Dahl, G. B. and Lochner, L. (2012), 'The impact of family income on child achievement: Evidence from the earned income tax credit', American Economic Review 102 (5), 1927-1956.
- de Chaisemartin, C. (2010), A note on instrumented difference in differences. Working Paper.
- de Chaisemartin, C. and D'Haultfœuille, X. (2018), 'Fuzzy differences-in-differences', The Review of Economic Studies 85 (2), 999-1028.
- Dinkelman, T. (2011), 'The effects of rural electrification on employment: New evidence from south africa', American Economic Review 101 (7), 3078-3108.
- Duflo, E. (2001), 'Schooling and labor market consequences of school construction in indonesia: Evidence from an unusual policy experiment', American Economic Review 91 (4), 795-813.
- Duggan, M. and Morton, F. S. (2010), 'The effect of medicare part d on pharmaceutical prices and utilization', American Economic Review 100 (1), 590-607.
- Duranton, G. and Turner, M. A. (2011), 'The fundamental law of road congestion: Evidence from us cities', American Economic Review 101 (6), 2616-2652.
- Ellul, A., Pagano, M. and Panunzi, F. (2010), 'Inheritance law and investment in family firms', American Economic Review 100 (5), 2414-2450.
- Enikolopov, R., Petrova, M. and Zhuravskaya, E. (2011), 'Media and political persuasion: Evidence from russia', American Economic Review 101 (7), 3253-3285.
- Fang, H. and Gavazza, A. (2011), 'Dynamic inefficiencies in an employment-based health insurance system: Theory and evidence', American Economic Review 101 (7), 3047-77.
- Faye, M. and Niehaus, P. (2012), 'Political aid cycles', American Economic Review 102 (7), 35163530.
- Forman, C., Goldfarb, A. and Greenstein, S. (2012), 'The internet and local wages: A puzzle', American Economic Review 102 (1), 556-575.
- Gentzkow, M., Shapiro, J. M. and Sinkinson, M. (2011), 'The effect of newspaper entry and exit on electoral politics', American Economic Review 101 (7), 2980-3018.
- Hornbeck, R. (2012), 'The enduring impact of the american dust bowl: Short-and long-run adjustments to environmental catastrophe', American Economic Review 102 (4), 1477-1507.

- Hotz, V. J. and Xiao, M. (2011), 'The impact of regulations on the supply and quality of care in child care markets', American Economic Review 101 (5), 1775-1805.
- Hudson, S., Hull, P. and Liebersohn, C. (2015), Interpreting instrumented difference-indifferences, Technical report, Working Paper (available upon request).
- Imbens, G. W. and Angrist, J. D. (1994), 'Identification and estimation of local average treatment effects', Econometrica 62 (2), 467-475.
- Imberman, S. A., Kugler, A. D. and Sacerdote, B. I. (2012), 'Katrina's children: Evidence on the structure of peer effects from hurricane evacuees', American Economic Review 102 (5), 20482082.
- Liu, R. Y. and Singh, K. (1995), 'Using iid bootstrap inference for general non-iid models', Journal of statistical planning and inference 43 (1-2), 67-75.
- Mian, A. and Sufi, A. (2011), 'House prices, home equity-based borrowing, and the us household leverage crisis', American Economic Review 101 (5), 2132-2156.
- Moser, P. and Voena, A. (2012), 'Compulsory licensing: Evidence from the trading with the enemy act', American Economic Review 102 (1), 396-427.
- Simcoe, T. (2012), 'Standard setting committees: Consensus governance for shared technology platforms', American Economic Review 102 (1), 305-336.
- van der Vaart, A. (2000), Asymptotics Statistics , Cambridge University Press.
- Wang, S.-Y. (2011), 'State misallocation and housing prices: Theory and evidence from china', American Economic Review 101 (5), 2081-2107.
- Zhang, X. M. and Zhu, F. (2011), 'Group size and incentives to contribute: A natural experiment at chinese wikipedia', American Economic Review 101 (4), 1601-1605.