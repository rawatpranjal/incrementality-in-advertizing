"""
Verify all numerical examples in sensitivity_analysis.tex
"""

import numpy as np

def verify_value(name, expected, actual, tolerance=0.01):
    """Verify a numerical value matches expected."""
    if abs(expected - actual) < tolerance:
        print(f"✓ {name}: {actual:.4f} (expected {expected:.4f})")
        return True
    else:
        print(f"✗ {name}: {actual:.4f} (expected {expected:.4f}) - MISMATCH!")
        return False

all_pass = True

print("="*80)
print("VERIFICATION OF NUMERICAL EXAMPLES")
print("="*80)

# Example 1: Problem Setting and Notation
print("\n[1] PROBLEM SETTING AND NOTATION")
print("-"*80)

n_exposed_disease = 120
n_exposed_total = 600
n_unexposed_disease = 50
n_unexposed_total = 500

p_d_e1 = n_exposed_disease / n_exposed_total
p_d_e0 = n_unexposed_disease / n_unexposed_total
rr_obs = p_d_e1 / p_d_e0

print(f"P(D=1|E=1, C=c) = {n_exposed_disease}/{n_exposed_total} = {p_d_e1:.4f}")
all_pass &= verify_value("P(D=1|E=1, C=c)", 0.20, p_d_e1)

print(f"P(D=1|E=0, C=c) = {n_unexposed_disease}/{n_unexposed_total} = {p_d_e0:.4f}")
all_pass &= verify_value("P(D=1|E=0, C=c)", 0.10, p_d_e0)

print(f"RR_obs = {p_d_e1:.4f} / {p_d_e0:.4f} = {rr_obs:.4f}")
all_pass &= verify_value("RR_obs", 2.0, rr_obs)

# Example 2: Defining Sensitivity Parameters
print("\n[2] DEFINING SENSITIVITY PARAMETERS")
print("-"*80)

# RR_EU calculation
p_u_e1 = np.array([0.20, 0.50, 0.30])
p_u_e0 = np.array([0.50, 0.40, 0.10])

rr_eu_levels = p_u_e1 / p_u_e0
print(f"RR_EU levels: {rr_eu_levels}")
all_pass &= verify_value("RR_EU(u1)", 0.40, rr_eu_levels[0])
all_pass &= verify_value("RR_EU(u2)", 1.25, rr_eu_levels[1])
all_pass &= verify_value("RR_EU(u3)", 3.00, rr_eu_levels[2])

rr_eu = np.max(rr_eu_levels)
print(f"RR_EU = max = {rr_eu:.4f}")
all_pass &= verify_value("RR_EU", 3.00, rr_eu)

# RR_UD calculation
p_d_e1_u = np.array([0.15, 0.20, 0.25])
p_d_e0_u = np.array([0.08, 0.10, 0.13])

rr_ud_values = []
# Within E=1
for i in range(3):
    for j in range(3):
        if i != j:
            rr = p_d_e1_u[i] / p_d_e1_u[j]
            rr_ud_values.append(rr)

# Within E=0
for i in range(3):
    for j in range(3):
        if i != j:
            rr = p_d_e0_u[i] / p_d_e0_u[j]
            rr_ud_values.append(rr)

rr_ud = np.max(rr_ud_values)
print(f"RR_UD = max of all ratios = {rr_ud:.4f}")
all_pass &= verify_value("RR_UD", 1.67, rr_ud, tolerance=0.01)

# Verify specific claim: P(D=1|E=1,u3)/P(D=1|E=1,u1) = 0.25/0.15 = 1.67
specific_ratio = p_d_e1_u[2] / p_d_e1_u[0]
print(f"P(D=1|E=1,u3)/P(D=1|E=1,u1) = {p_d_e1_u[2]:.2f}/{p_d_e1_u[0]:.2f} = {specific_ratio:.4f}")
all_pass &= verify_value("Specific max ratio", 1.67, specific_ratio, tolerance=0.01)

# Example 3: Bounding Factor
print("\n[3] BOUNDING FACTOR")
print("-"*80)

numerator = rr_eu * rr_ud
denominator = rr_eu + rr_ud - 1
bf = numerator / denominator

print(f"Numerator = RR_EU × RR_UD = {rr_eu:.2f} × {rr_ud:.2f} = {numerator:.4f}")
all_pass &= verify_value("Numerator", 5.00, numerator, tolerance=0.02)

print(f"Denominator = RR_EU + RR_UD - 1 = {rr_eu:.2f} + {rr_ud:.2f} - 1 = {denominator:.4f}")
all_pass &= verify_value("Denominator", 3.67, denominator, tolerance=0.01)

print(f"BF = {numerator:.4f} / {denominator:.4f} = {bf:.4f}")
all_pass &= verify_value("Bounding Factor", 1.36, bf, tolerance=0.01)

rr_true_lower = rr_obs / bf
print(f"RR_true >= {rr_obs:.2f} / {bf:.2f} = {rr_true_lower:.4f}")
all_pass &= verify_value("RR_true lower bound", 1.47, rr_true_lower, tolerance=0.01)

# Example 4: E-value
print("\n[4] E-VALUE")
print("-"*80)

e_value = rr_obs + np.sqrt(rr_obs * (rr_obs - 1))
sqrt_term = np.sqrt(rr_obs * (rr_obs - 1))

print(f"sqrt(RR × (RR-1)) = sqrt({rr_obs:.2f} × {rr_obs - 1:.2f}) = sqrt({rr_obs * (rr_obs - 1):.4f}) = {sqrt_term:.4f}")
all_pass &= verify_value("sqrt term", 1.41, sqrt_term, tolerance=0.01)

print(f"E-value = {rr_obs:.2f} + {sqrt_term:.2f} = {e_value:.4f}")
all_pass &= verify_value("E-value", 3.41, e_value, tolerance=0.01)

# Example 5: E-value Formula Derivation
print("\n[5] E-VALUE FORMULA DERIVATION")
print("-"*80)

a = 1
b = -2 * rr_obs
c = rr_obs

print(f"Quadratic: ax² + bx + c = 0")
print(f"a = {a}")
print(f"b = -2 × RR_obs = -2 × {rr_obs:.2f} = {b:.4f}")
all_pass &= verify_value("b coefficient", -4.00, b)

print(f"c = RR_obs = {c:.4f}")
all_pass &= verify_value("c coefficient", 2.00, c)

discriminant = b**2 - 4*a*c
print(f"discriminant = b² - 4ac = {b:.2f}² - 4({a})({c:.2f}) = {b**2:.4f} - {4*a*c:.4f} = {discriminant:.4f}")
all_pass &= verify_value("discriminant", 8.00, discriminant)

sqrt_discriminant = np.sqrt(discriminant)
print(f"sqrt(discriminant) = sqrt({discriminant:.2f}) = {sqrt_discriminant:.4f}")
all_pass &= verify_value("sqrt(discriminant)", 2.83, sqrt_discriminant, tolerance=0.01)

e_value_from_quad = (-b + sqrt_discriminant) / (2*a)
print(f"E-value = (-b + sqrt(discriminant)) / (2a)")
print(f"        = ({-b:.2f} + {sqrt_discriminant:.2f}) / {2*a}")
print(f"        = {-b + sqrt_discriminant:.4f} / {2*a}")
print(f"        = {e_value_from_quad:.4f}")
all_pass &= verify_value("E-value from quadratic", 3.41, e_value_from_quad, tolerance=0.01)

print(f"Matches closed form: {e_value:.4f}")
all_pass &= verify_value("E-values match", e_value, e_value_from_quad, tolerance=0.001)

# Example 6: High Threshold Condition
print("\n[6] HIGH THRESHOLD CONDITION")
print("-"*80)

# Case 1: Cornfield conditions met (RR_EU = RR_UD = 2.0)
rr_eu_test = 2.0
rr_ud_test = 2.0

bf_test = (rr_eu_test * rr_ud_test) / (rr_eu_test + rr_ud_test - 1)
print(f"Case 1: RR_EU = RR_UD = {rr_eu_test:.1f}")
print(f"BF = ({rr_eu_test:.1f} × {rr_ud_test:.1f}) / ({rr_eu_test:.1f} + {rr_ud_test:.1f} - 1)")
print(f"   = {rr_eu_test * rr_ud_test:.1f} / {rr_eu_test + rr_ud_test - 1:.1f}")
print(f"   = {bf_test:.4f}")
all_pass &= verify_value("BF (Cornfield)", 1.33, bf_test, tolerance=0.01)

rr_true_test = rr_obs / bf_test
print(f"RR_true >= {rr_obs:.2f} / {bf_test:.2f} = {rr_true_test:.4f}")
all_pass &= verify_value("RR_true (Cornfield)", 1.50, rr_true_test, tolerance=0.01)

# Case 2: E-value threshold met (RR_EU = RR_UD = 3.414)
rr_eu_evalue = 3.414
rr_ud_evalue = 3.414

bf_evalue = (rr_eu_evalue * rr_ud_evalue) / (rr_eu_evalue + rr_ud_evalue - 1)
numerator_evalue = rr_eu_evalue * rr_ud_evalue
denominator_evalue = rr_eu_evalue + rr_ud_evalue - 1

print(f"\nCase 2: RR_EU = RR_UD = {rr_eu_evalue:.3f}")
print(f"BF = ({rr_eu_evalue:.3f} × {rr_ud_evalue:.3f}) / ({rr_eu_evalue:.3f} + {rr_ud_evalue:.3f} - 1)")
print(f"   = {numerator_evalue:.4f} / {denominator_evalue:.4f}")
print(f"   = {bf_evalue:.4f}")
all_pass &= verify_value("Numerator (E-value)", 11.655, numerator_evalue, tolerance=0.01)
all_pass &= verify_value("Denominator (E-value)", 5.828, denominator_evalue, tolerance=0.01)
all_pass &= verify_value("BF (E-value)", 2.00, bf_evalue, tolerance=0.01)

rr_true_evalue = rr_obs / bf_evalue
print(f"RR_true >= {rr_obs:.2f} / {bf_evalue:.2f} = {rr_true_evalue:.4f}")
all_pass &= verify_value("RR_true (E-value)", 1.00, rr_true_evalue, tolerance=0.01)

# Summary
print("\n" + "="*80)
if all_pass:
    print("✓ ALL NUMERICAL EXAMPLES VERIFIED SUCCESSFULLY")
else:
    print("✗ SOME NUMERICAL EXAMPLES FAILED VERIFICATION")
print("="*80)
