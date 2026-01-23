"""
Compute numerical examples for sensitivity analysis appendix.
Outputs all results to sensitivity_examples.txt.
"""

import numpy as np

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80 + "\n")

def print_subsection(title):
    """Print a subsection header."""
    print("\n" + "-"*80)
    print(f"{title}")
    print("-"*80 + "\n")

# Example 1: Problem Setting and Notation
print_section("EXAMPLE 1: PROBLEM SETTING AND NOTATION")

print("Consider a hypothetical study examining the effect of a drug (E) on")
print("a disease outcome (D), adjusted for age group (C).")
print()
print("Within age group C=c, we observe:")
print("  - Among exposed (E=1): 120 out of 600 developed disease")
print("  - Among unexposed (E=0): 50 out of 500 developed disease")
print()

n_exposed_disease = 120
n_exposed_total = 600
n_unexposed_disease = 50
n_unexposed_total = 500

p_d_given_e1 = n_exposed_disease / n_exposed_total
p_d_given_e0 = n_unexposed_disease / n_unexposed_total
rr_obs = p_d_given_e1 / p_d_given_e0

print(f"P(D=1|E=1, C=c) = {n_exposed_disease}/{n_exposed_total} = {p_d_given_e1:.4f}")
print(f"P(D=1|E=0, C=c) = {n_unexposed_disease}/{n_unexposed_total} = {p_d_given_e0:.4f}")
print()
print(f"Observed risk ratio: RR_ED^obs = {p_d_given_e1:.4f} / {p_d_given_e0:.4f} = {rr_obs:.3f}")
print()
print("This observed association of RR=2.0 could be causal, or it could be")
print("explained (partially or fully) by an unmeasured confounder U.")

# Example 2: Defining Sensitivity Parameters
print_section("EXAMPLE 2: DEFINING SENSITIVITY PARAMETERS")

print("Suppose the unmeasured confounder U has 3 levels (u1, u2, u3).")
print("We posit the following distribution and disease risks:")
print()

# Distribution of U by exposure status
print("Distribution of U by exposure:")
print("  Level    P(U|E=1)    P(U|E=0)")
print("  u1       0.20        0.50")
print("  u2       0.50        0.40")
print("  u3       0.30        0.10")
print()

p_u_given_e1 = np.array([0.20, 0.50, 0.30])
p_u_given_e0 = np.array([0.50, 0.40, 0.10])

# Calculate RR_EU for each level
print("RR_EU for each level:")
rr_eu_levels = []
for i in range(3):
    rr = p_u_given_e1[i] / p_u_given_e0[i]
    rr_eu_levels.append(rr)
    print(f"  u{i+1}: P(U=u{i+1}|E=1) / P(U=u{i+1}|E=0) = {p_u_given_e1[i]:.2f} / {p_u_given_e0[i]:.2f} = {rr:.2f}")

rr_eu = max(rr_eu_levels)
print()
print(f"RR_EU = max(RR_EU for all levels) = {rr_eu:.2f}")
print()

# Disease risks by U and E
print("Disease risks P(D=1|E,U):")
print("  Level    P(D=1|E=1,U)    P(D=1|E=0,U)")
print("  u1       0.15            0.08")
print("  u2       0.20            0.10")
print("  u3       0.25            0.13")
print()

p_d_e1_u = np.array([0.15, 0.20, 0.25])
p_d_e0_u = np.array([0.08, 0.10, 0.13])

# Calculate RR_UD - maximum across all comparisons
print("RR_UD is the maximum ratio comparing any two levels of U,")
print("within either exposure group:")
print()

rr_ud_values = []
for e_idx, (p_d_u, e_label) in enumerate([(p_d_e1_u, "E=1"), (p_d_e0_u, "E=0")]):
    print(f"Within {e_label}:")
    for i in range(3):
        for j in range(3):
            if i != j:
                rr = p_d_u[i] / p_d_u[j]
                rr_ud_values.append(rr)
                print(f"  P(D=1|{e_label},u{i+1}) / P(D=1|{e_label},u{j+1}) = {p_d_u[i]:.2f} / {p_d_u[j]:.2f} = {rr:.3f}")

rr_ud = max(rr_ud_values)
print()
print(f"RR_UD = max(all ratios) = {rr_ud:.3f}")

# Example 3: Derivation of Bounding Factor
print_section("EXAMPLE 3: DERIVATION OF BOUNDING FACTOR")

print(f"Using the observed RR_ED^obs = {rr_obs:.3f} from Example 1,")
print(f"and the sensitivity parameters from Example 2:")
print(f"  RR_EU = {rr_eu:.2f}")
print(f"  RR_UD = {rr_ud:.3f}")
print()

# Calculate bounding factor
numerator = rr_eu * rr_ud
denominator = rr_eu + rr_ud - 1
bounding_factor = numerator / denominator

print("The bounding factor is:")
print(f"  BF = (RR_EU × RR_UD) / (RR_EU + RR_UD - 1)")
print(f"     = ({rr_eu:.2f} × {rr_ud:.3f}) / ({rr_eu:.2f} + {rr_ud:.3f} - 1)")
print(f"     = {numerator:.3f} / {denominator:.3f}")
print(f"     = {bounding_factor:.3f}")
print()

# Calculate lower bound for true RR
rr_true_lower_bound = rr_obs / bounding_factor

print("The bias-corrected lower bound for the true causal risk ratio is:")
print(f"  RR_ED^true ≥ RR_ED^obs / BF")
print(f"             = {rr_obs:.3f} / {bounding_factor:.3f}")
print(f"             = {rr_true_lower_bound:.3f}")
print()
print("Interpretation: Even in the presence of an unmeasured confounder")
print(f"with strengths RR_EU={rr_eu:.2f} and RR_UD={rr_ud:.3f}, the true causal")
print(f"effect cannot be less than {rr_true_lower_bound:.3f}. The association remains")
print("substantially elevated above the null (RR=1).")

# Example 4: Defining the E-Value
print_section("EXAMPLE 4: DEFINING THE E-VALUE")

print(f"Given an observed risk ratio of RR_ED^obs = {rr_obs:.3f},")
print("we ask: What is the minimum strength of association that")
print("an unmeasured confounder would need to have with both the")
print("exposure and outcome to fully explain away this association?")
print()

# Calculate E-value
e_value = rr_obs + np.sqrt(rr_obs * (rr_obs - 1))

print("The E-value formula gives:")
print(f"  E-value = RR + sqrt(RR × (RR - 1))")
print(f"          = {rr_obs:.3f} + sqrt({rr_obs:.3f} × ({rr_obs:.3f} - 1))")
print(f"          = {rr_obs:.3f} + sqrt({rr_obs * (rr_obs - 1):.3f})")
print(f"          = {rr_obs:.3f} + {np.sqrt(rr_obs * (rr_obs - 1)):.3f}")
print(f"          = {e_value:.3f}")
print()
print("Interpretation: To explain away the observed RR=2.0 as entirely")
print(f"due to confounding, an unmeasured confounder would need to be")
print(f"associated with both the exposure and outcome by a risk ratio")
print(f"of {e_value:.3f}-fold each, above and beyond the measured covariates.")

# Example 5: E-Value Formula Derivation with Specific Numbers
print_section("EXAMPLE 5: E-VALUE FORMULA DERIVATION (WORKED EXAMPLE)")

print(f"We solve the quadratic equation for RR_ED^obs = {rr_obs:.3f}.")
print()
print("Starting from the condition that RR_EU = RR_UD = E-value,")
print("and the bounding factor equals the observed RR:")
print()
print("  (E-value)² / (2·E-value - 1) = RR_ED^obs")
print()
print("Rearranging:")
print(f"  (E-value)² = {rr_obs:.3f} × (2·E-value - 1)")
print(f"  (E-value)² = {2*rr_obs:.3f}·E-value - {rr_obs:.3f}")
print(f"  (E-value)² - {2*rr_obs:.3f}·E-value + {rr_obs:.3f} = 0")
print()

a = 1
b = -2 * rr_obs
c = rr_obs

print("This is a quadratic equation ax² + bx + c = 0 where:")
print(f"  a = {a}")
print(f"  b = {b:.3f}")
print(f"  c = {c:.3f}")
print()

discriminant = b**2 - 4*a*c
print("Using the quadratic formula:")
print(f"  discriminant = b² - 4ac = ({b:.3f})² - 4({a})({c:.3f})")
print(f"               = {b**2:.3f} - {4*a*c:.3f}")
print(f"               = {discriminant:.3f}")
print()

sqrt_discriminant = np.sqrt(discriminant)
e_value_calc = (-b + sqrt_discriminant) / (2*a)

print(f"  E-value = (-b + sqrt(discriminant)) / (2a)")
print(f"          = ({-b:.3f} + sqrt({discriminant:.3f})) / {2*a}")
print(f"          = ({-b:.3f} + {sqrt_discriminant:.3f}) / {2*a}")
print(f"          = {-b + sqrt_discriminant:.3f} / {2*a}")
print(f"          = {e_value_calc:.3f}")
print()
print("Which matches the closed-form E-value formula:")
print(f"  E-value = RR + sqrt(RR(RR-1)) = {e_value:.3f}")

# Example 6: High Threshold Condition
print_section("EXAMPLE 6: HIGH THRESHOLD CONDITION")

print(f"For the observed RR_ED^obs = {rr_obs:.3f}, we compare:")
print()

print("Classic Cornfield conditions:")
print("  To explain away the effect, both:")
print(f"    RR_EU ≥ {rr_obs:.3f}")
print(f"    RR_UD ≥ {rr_obs:.3f}")
print()

print("High threshold condition (stronger):")
print(f"  To explain away the effect:")
print(f"    max(RR_EU, RR_UD) ≥ E-value = {e_value:.3f}")
print()

print("Numerical demonstration:")
print("Suppose RR_EU = RR_UD = 2.0 (meeting Cornfield conditions).")
print()

rr_eu_test = 2.0
rr_ud_test = 2.0

bf_test = (rr_eu_test * rr_ud_test) / (rr_eu_test + rr_ud_test - 1)
rr_true_test = rr_obs / bf_test

print(f"  Bounding factor = ({rr_eu_test:.1f} × {rr_ud_test:.1f}) / ({rr_eu_test:.1f} + {rr_ud_test:.1f} - 1)")
print(f"                  = {rr_eu_test * rr_ud_test:.1f} / {rr_eu_test + rr_ud_test - 1:.1f}")
print(f"                  = {bf_test:.3f}")
print()
print(f"  RR_ED^true ≥ {rr_obs:.3f} / {bf_test:.3f} = {rr_true_test:.3f}")
print()
print(f"The effect is NOT explained away (RR^true = {rr_true_test:.3f} > 1).")
print()

print(f"Now suppose max(RR_EU, RR_UD) = E-value = {e_value:.3f}.")
print(f"Let RR_EU = RR_UD = {e_value:.3f}.")
print()

rr_eu_evalue = e_value
rr_ud_evalue = e_value

bf_evalue = (rr_eu_evalue * rr_ud_evalue) / (rr_eu_evalue + rr_ud_evalue - 1)
rr_true_evalue = rr_obs / bf_evalue

print(f"  Bounding factor = ({rr_eu_evalue:.3f} × {rr_ud_evalue:.3f}) / ({rr_eu_evalue:.3f} + {rr_ud_evalue:.3f} - 1)")
print(f"                  = {rr_eu_evalue * rr_ud_evalue:.3f} / {rr_eu_evalue + rr_ud_evalue - 1:.3f}")
print(f"                  = {bf_evalue:.3f}")
print()
print(f"  RR_ED^true ≥ {rr_obs:.3f} / {bf_evalue:.3f} = {rr_true_evalue:.3f}")
print()
print("The effect is now explained away (RR^true ≈ 1).")
print()
print(f"Conclusion: The high threshold condition (max ≥ {e_value:.3f}) is indeed")
print(f"stronger than the Cornfield conditions (both ≥ {rr_obs:.3f}).")

print("\n" + "="*80)
print("END OF EXAMPLES")
print("="*80)
