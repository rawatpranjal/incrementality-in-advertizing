"""
Common Data Generating Process for Ghost Ads Simulations

Implements the potential outcomes framework with:
- Potential outcomes Y_i(0), Y_i(1)
- Random assignment Z_i
- Actual exposure D_i with compliance rate π
- Ghost ad indicators GA_i
- Predicted exposure D̂_i
- Placebo exposure indicators for PSA

All estimators (ITT, LATE, PSA, Ghost Ads, PGA) are implemented here.
"""

import numpy as np


class GhostAdsDGP:
    """
    Data Generating Process for Ghost Ads simulations.

    Parameters:
    -----------
    N : int
        Sample size
    pi : float
        Compliance rate P(D_i=1 | Z_i=1)
    tau : float
        True treatment effect (constant across all units)
    sigma_0 : float
        Standard deviation of Y_i(0)
    sigma_1 : float
        Standard deviation of Y_i(1)
    p_predict : float
        Prediction accuracy for PGA (probability that D̂_i = D_i(1))
    seed : int
        Random seed for reproducibility
    """

    def __init__(self, N=1000, pi=0.5, tau=2.0, sigma_0=2.0, sigma_1=None,
                 p_predict=0.97, seed=None):
        self.N = N
        self.pi = pi
        self.tau = tau
        self.sigma_0 = sigma_0
        # Note: sigma_1 is not used in this DGP because treatment effects are constant.
        # With constant effects τ_i = τ for all i, we have Y_i(1) = Y_i(0) + τ,
        # which implies Var[Y_i(1)] = Var[Y_i(0)] = sigma_0^2.
        # The parameter is kept for API compatibility but defaults to None.
        if sigma_1 is not None and sigma_1 != sigma_0:
            import warnings
            warnings.warn(f"sigma_1={sigma_1} is ignored. With constant treatment effects, "
                          f"Var[Y(1)] = Var[Y(0)] = sigma_0^2 = {sigma_0**2}")
        self.sigma_1 = sigma_0  # Always equals sigma_0 under constant effects
        self.p_predict = p_predict
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self._generate_data()

    def _generate_data(self):
        """Generate potential outcomes and treatment assignment."""
        # Potential outcomes
        # Y_i(0) ~ N(0, sigma_0^2)
        # Y_i(1) = Y_i(0) + tau (constant treatment effect)
        # Note: This implies Var[Y_i(1)] = Var[Y_i(0)] = sigma_0^2
        self.Y0 = np.random.normal(0, self.sigma_0, self.N)
        self.Y1 = self.Y0 + self.tau

        # Individual treatment effects (constant in this DGP)
        self.tau_i = np.full(self.N, self.tau)

        # Random assignment: Z_i ~ Bernoulli(0.5)
        self.Z = np.random.binomial(1, 0.5, self.N)

        # Actual exposure with one-sided noncompliance
        # D_i(0) = 0 for all i
        # D_i(1) ~ Bernoulli(pi) for treatment arm
        self.D_potential_1 = np.random.binomial(1, self.pi, self.N)
        self.D_potential_0 = np.zeros(self.N, dtype=int)

        # Observed exposure: D_i = Z_i * D_i(1)
        self.D = self.Z * self.D_potential_1

        # Ghost ad indicators for control users
        # GA_i = 1 ⟺ D_i(T) = 1 (counterfactual tagging)
        # This is only relevant for control users (Z_i = 0)
        self.GA = (1 - self.Z) * self.D_potential_1

        # Predicted exposure D̂_i for PGA
        # Pre-randomization prediction with accuracy p_predict
        # D̂_i ⊥ Z_i (must be independent of assignment)
        # With probability p_predict: D̂_i = D_i(1) (correct prediction)
        # With probability (1-p_predict): D̂_i = 1 - D_i(1) (incorrect prediction)
        prediction_error = np.random.binomial(1, 1 - self.p_predict, self.N)
        self.D_hat = np.where(prediction_error == 1,
                               1 - self.D_potential_1,  # Wrong prediction
                               self.D_potential_1)       # Correct prediction

        # Placebo exposure for PSA design
        # Perfect blind: D_i(CP) = P ⟺ D_i(T) = 1
        # We'll create a separate assignment for PSA
        self.Z_psa = np.random.binomial(1, 0.5, self.N)  # T vs CP
        self.D_psa = np.where(self.Z_psa == 1,
                               self.D_potential_1,  # Real ad in treatment
                               self.D_potential_1)  # Placebo in control (P indicator)

        # Observed outcome: Y_i = Y_i(0) + D_i * tau_i
        # For PSA: Y_i(P) = Y_i(0) (no placebo effect)
        self.Y_obs = self.Y0 + self.D * self.tau_i
        self.Y_obs_psa = self.Y0 + self.Z_psa * self.D_psa * self.tau_i

    def estimate_itt(self):
        """
        ITT Estimator: τ̂_ITT = Ȳ_1 - Ȳ_0

        Returns:
        --------
        dict with keys: estimate, se, n_treated, n_control
        """
        Y1 = self.Y_obs[self.Z == 1]
        Y0 = self.Y_obs[self.Z == 0]

        n1 = len(Y1)
        n0 = len(Y0)

        tau_itt = np.mean(Y1) - np.mean(Y0)

        # Conservative variance: s²_1/n_1 + s²_0/n_0
        var_itt = np.var(Y1, ddof=1) / n1 + np.var(Y0, ddof=1) / n0
        se_itt = np.sqrt(var_itt)

        return {
            'estimate': tau_itt,
            'se': se_itt,
            'variance': var_itt,
            'n_treated': n1,
            'n_control': n0
        }

    def estimate_late(self):
        """
        LATE Estimator (Wald): τ̂_LATE = (Ȳ_1 - Ȳ_0) / (D̄_1 - D̄_0)

        Returns:
        --------
        dict with keys: estimate, se, first_stage, n_treated, n_control
        """
        Y1 = self.Y_obs[self.Z == 1]
        Y0 = self.Y_obs[self.Z == 0]
        D1 = self.D[self.Z == 1]
        D0 = self.D[self.Z == 0]

        n1 = len(Y1)
        n0 = len(Y0)

        # Reduced form
        beta_hat = np.mean(Y1) - np.mean(Y0)

        # First stage
        pi_hat = np.mean(D1) - np.mean(D0)

        # Wald estimator
        if pi_hat == 0:
            return {
                'estimate': np.nan,
                'se': np.nan,
                'variance': np.nan,
                'first_stage': pi_hat,
                'n_treated': n1,
                'n_control': n0
            }

        tau_late = beta_hat / pi_hat

        # Delta method variance (simplified)
        var_beta = np.var(Y1, ddof=1) / n1 + np.var(Y0, ddof=1) / n0
        var_pi = np.var(D1, ddof=1) / n1 + np.var(D0, ddof=1) / n0

        # Approximate variance (ignoring covariance term)
        var_late = (var_beta + tau_late**2 * var_pi) / pi_hat**2
        se_late = np.sqrt(var_late)

        return {
            'estimate': tau_late,
            'se': se_late,
            'variance': var_late,
            'first_stage': pi_hat,
            'n_treated': n1,
            'n_control': n0
        }

    def estimate_psa(self):
        """
        PSA Estimator: τ̂_PSA = Ȳ_{T,D=1} - Ȳ_{CP,D=P}

        Uses the PSA-specific assignment Z_psa where:
        - Z_psa = 1: Treatment arm (real ad)
        - Z_psa = 0: Control-Placebo arm (placebo ad)

        Returns:
        --------
        dict with keys: estimate, se, n_treated_exposed, n_control_placebo
        """
        # Treatment arm: exposed users
        treated_exposed = (self.Z_psa == 1) & (self.D_psa == 1)
        Y_t1 = self.Y_obs_psa[treated_exposed]

        # Control-placebo arm: placebo-exposed users
        control_placebo = (self.Z_psa == 0) & (self.D_psa == 1)
        Y_cp = self.Y_obs_psa[control_placebo]

        nt1 = len(Y_t1)
        ncp = len(Y_cp)

        if nt1 == 0 or ncp == 0:
            return {
                'estimate': np.nan,
                'se': np.nan,
                'variance': np.nan,
                'n_treated_exposed': nt1,
                'n_control_placebo': ncp
            }

        tau_psa = np.mean(Y_t1) - np.mean(Y_cp)

        # Conservative variance
        var_psa = np.var(Y_t1, ddof=1) / nt1 + np.var(Y_cp, ddof=1) / ncp
        se_psa = np.sqrt(var_psa)

        return {
            'estimate': tau_psa,
            'se': se_psa,
            'variance': var_psa,
            'n_treated_exposed': nt1,
            'n_control_placebo': ncp
        }

    def estimate_ghost_ads(self):
        """
        Ghost Ads Estimator: θ̂_GA = Ȳ_{T,D=1} - Ȳ_{C,GA=1}

        Returns:
        --------
        dict with keys: estimate, se, n_treated_exposed, n_control_ghost
        """
        # Treatment arm: exposed users
        treated_exposed = (self.Z == 1) & (self.D == 1)
        Y_t1 = self.Y_obs[treated_exposed]

        # Control arm: ghost ad tagged users
        control_ghost = (self.Z == 0) & (self.GA == 1)
        Y_cga = self.Y_obs[control_ghost]

        nt1 = len(Y_t1)
        ncga = len(Y_cga)

        if nt1 == 0 or ncga == 0:
            return {
                'estimate': np.nan,
                'se': np.nan,
                'variance': np.nan,
                'n_treated_exposed': nt1,
                'n_control_ghost': ncga
            }

        theta_ga = np.mean(Y_t1) - np.mean(Y_cga)

        # Conservative variance
        var_ga = np.var(Y_t1, ddof=1) / nt1 + np.var(Y_cga, ddof=1) / ncga
        se_ga = np.sqrt(var_ga)

        return {
            'estimate': theta_ga,
            'se': se_ga,
            'variance': var_ga,
            'n_treated_exposed': nt1,
            'n_control_ghost': ncga
        }

    def estimate_pga(self):
        """
        PGA Estimator: τ̂_PGA = (Ȳ_{T,D̂=1} - Ȳ_{C,D̂=1}) / p̂

        where p̂ = P(D=1 | Z=1, D̂=1)

        Returns:
        --------
        dict with keys: estimate, se, compliance_rate, n_treated_predicted, n_control_predicted
        """
        # Treatment arm: predicted-exposed users
        treated_predicted = (self.Z == 1) & (self.D_hat == 1)
        Y_t_dhat = self.Y_obs[treated_predicted]
        D_t_dhat = self.D[treated_predicted]

        # Control arm: predicted-exposed users
        control_predicted = (self.Z == 0) & (self.D_hat == 1)
        Y_c_dhat = self.Y_obs[control_predicted]

        nt_dhat = len(Y_t_dhat)
        nc_dhat = len(Y_c_dhat)

        if nt_dhat == 0 or nc_dhat == 0:
            return {
                'estimate': np.nan,
                'se': np.nan,
                'variance': np.nan,
                'compliance_rate': np.nan,
                'n_treated_predicted': nt_dhat,
                'n_control_predicted': nc_dhat
            }

        # Reduced form on predicted-exposed
        beta_pga = np.mean(Y_t_dhat) - np.mean(Y_c_dhat)

        # Compliance rate among predicted-exposed in treatment
        p_hat = np.mean(D_t_dhat) if nt_dhat > 0 else 0

        if p_hat == 0:
            return {
                'estimate': np.nan,
                'se': np.nan,
                'variance': np.nan,
                'compliance_rate': p_hat,
                'n_treated_predicted': nt_dhat,
                'n_control_predicted': nc_dhat
            }

        tau_pga = beta_pga / p_hat

        # Delta method variance (simplified)
        var_beta = np.var(Y_t_dhat, ddof=1) / nt_dhat + np.var(Y_c_dhat, ddof=1) / nc_dhat
        var_p = p_hat * (1 - p_hat) / nt_dhat

        # Approximate variance (ignoring covariance)
        var_pga = (var_beta + tau_pga**2 * var_p) / p_hat**2
        se_pga = np.sqrt(var_pga)

        return {
            'estimate': tau_pga,
            'se': se_pga,
            'variance': var_pga,
            'compliance_rate': p_hat,
            'n_treated_predicted': nt_dhat,
            'n_control_predicted': nc_dhat
        }

    def get_true_effects(self):
        """
        Calculate true treatment effects.

        Returns:
        --------
        dict with keys: tau_att, tau_itt, tau_ate
        """
        # ATT: average effect on treated (exposed)
        exposed = self.D_potential_1 == 1
        tau_att = np.mean(self.tau_i[exposed])

        # ITT: pi * ATT (under one-sided noncompliance)
        tau_itt = self.pi * tau_att

        # ATE: average effect over all units
        tau_ate = np.mean(self.tau_i)

        return {
            'tau_att': tau_att,
            'tau_itt': tau_itt,
            'tau_ate': tau_ate,
            'pi': self.pi
        }


def monte_carlo_simulation(N, pi, tau, sigma_0, sigma_1, n_sims=10000,
                           p_predict=0.97, seed=42):
    """
    Run Monte Carlo simulation for all estimators.

    Parameters:
    -----------
    N : int
        Sample size
    pi : float
        Compliance rate
    tau : float
        True treatment effect
    sigma_0, sigma_1 : float
        Outcome standard deviations
    n_sims : int
        Number of Monte Carlo iterations
    p_predict : float
        Prediction accuracy for PGA
    seed : int
        Random seed

    Returns:
    --------
    dict with results for each estimator
    """
    np.random.seed(seed)

    results = {
        'itt': {'estimates': [], 'variances': []},
        'late': {'estimates': [], 'variances': [], 'first_stages': []},
        'psa': {'estimates': [], 'variances': []},
        'ga': {'estimates': [], 'variances': []},
        'pga': {'estimates': [], 'variances': [], 'compliance_rates': []}
    }

    for i in range(n_sims):
        # Generate new data each iteration
        dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0, sigma_1=sigma_1,
                          p_predict=p_predict, seed=seed + i)

        # ITT
        itt_res = dgp.estimate_itt()
        results['itt']['estimates'].append(itt_res['estimate'])
        results['itt']['variances'].append(itt_res['variance'])

        # LATE
        late_res = dgp.estimate_late()
        results['late']['estimates'].append(late_res['estimate'])
        results['late']['variances'].append(late_res['variance'])
        results['late']['first_stages'].append(late_res['first_stage'])

        # PSA
        psa_res = dgp.estimate_psa()
        results['psa']['estimates'].append(psa_res['estimate'])
        results['psa']['variances'].append(psa_res['variance'])

        # Ghost Ads
        ga_res = dgp.estimate_ghost_ads()
        results['ga']['estimates'].append(ga_res['estimate'])
        results['ga']['variances'].append(ga_res['variance'])

        # PGA
        pga_res = dgp.estimate_pga()
        results['pga']['estimates'].append(pga_res['estimate'])
        results['pga']['variances'].append(pga_res['variance'])
        results['pga']['compliance_rates'].append(pga_res['compliance_rate'])

    # Convert to numpy arrays
    for estimator in results:
        for key in results[estimator]:
            results[estimator][key] = np.array(results[estimator][key])

    # True effects
    dgp = GhostAdsDGP(N=N, pi=pi, tau=tau, sigma_0=sigma_0, sigma_1=sigma_1,
                      p_predict=p_predict, seed=seed)
    true_effects = dgp.get_true_effects()
    results['true_effects'] = true_effects

    return results
