import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Create output directory if it doesn't exist
output_dir = '/Users/pranjal/Code/marketplace-incrementality/latex/appendix/figures'
os.makedirs(output_dir, exist_ok=True)

# Define decay functions
def exponential_decay(t, lambda_param=0.15):
    """Exponential decay function"""
    return np.exp(-lambda_param * t)

def gamma_decay(t, k=3, theta=5):
    """Gamma decay function - shows build-up then decay"""
    # Properly normalize at maximum, not at t=0
    max_val = stats.gamma.pdf((k-1)*theta, a=k, scale=theta) if k > 1 else stats.gamma.pdf(0, a=k, scale=theta)
    return stats.gamma.pdf(t, a=k, scale=theta) / max_val

def weibull_decay(t, k=0.7, lambda_param=10):
    """Weibull decay function - shows rapid initial decay"""
    # Normalize at t=0
    return np.exp(-np.power(t/lambda_param, k))

def uniform_decay(t, T=14):
    """Uniform (boxcar) decay function"""
    return np.where(t <= T, 1.0, 0.0)

# Plot 1: Comparison of Decay Kernels
def plot_decay_kernels():
    fig, ax = plt.subplots(figsize=(8, 5))

    t = np.linspace(0, 30, 500)

    # Plot different kernels with more distinct parameters
    ax.plot(t, exponential_decay(t, 0.15), label='Exponential ($\\lambda=0.15$)', linewidth=2.5, color='#1f77b4')
    ax.plot(t, gamma_decay(t, 3, 5), label='Gamma ($k=3, \\theta=5$) - Build-up', linewidth=2.5, linestyle='--', color='#ff7f0e')
    ax.plot(t, weibull_decay(t, 0.7, 10), label='Weibull ($k=0.7, \\lambda=10$) - Fast decay', linewidth=2.5, linestyle='-.', color='#2ca02c')
    ax.plot(t, uniform_decay(t, 14), label='Uniform/Boxcar ($T=14$ days)', linewidth=2.5, linestyle=':', color='#d62728')

    ax.set_xlabel('Time Since Impression (days)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Relative Ad Effect $f(\\Delta t)$', fontweight='bold', fontsize=13)
    ax.set_title('Comparison of Ad Stock Decay Functions', fontweight='bold', pad=15, fontsize=14)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.1, which='minor', linestyle=':')
    ax.minorticks_on()
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.05, 1.15)

    # Add annotations for key differences
    ax.annotate('Gamma: Initial build-up', xy=(10, 0.95), xytext=(12, 0.7),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', alpha=0.7),
                fontsize=10, color='#ff7f0e')
    ax.annotate('Weibull: Rapid initial decay', xy=(2, 0.5), xytext=(4, 0.25),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.7),
                fontsize=10, color='#2ca02c')
    ax.annotate('Uniform: Constant then zero', xy=(14, 0.5), xytext=(16, 0.65),
                arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.7),
                fontsize=10, color='#d62728')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/decay_kernels.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/decay_kernels.png', bbox_inches='tight')
    plt.close()

# Plot 2: Ad Stock Accumulation Example
def plot_adstock_accumulation():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Timeline
    t = np.linspace(0, 100, 1000)

    # Ad exposure times
    ad_times = [10, 25, 40, 55, 70, 85]
    ad_weights = [1, 1, 2, 1, 1, 1]  # Weight for each ad (e.g., 2 = two impressions)

    # Calculate adstock for exponential decay
    adstock = np.zeros_like(t)
    lambda_param = 0.05

    for ad_time, weight in zip(ad_times, ad_weights):
        mask = t >= ad_time
        adstock[mask] += weight * exponential_decay(t[mask] - ad_time, lambda_param)

    # Top panel: Individual ad impacts
    ax1.set_title('Individual Ad Exposures and Their Decay', fontweight='bold', pad=10)
    colors = plt.cm.Set2(np.linspace(0, 1, len(ad_times)))

    for i, (ad_time, weight, color) in enumerate(zip(ad_times, ad_weights, colors)):
        mask = t >= ad_time
        individual_impact = np.zeros_like(t)
        individual_impact[mask] = weight * exponential_decay(t[mask] - ad_time, lambda_param)
        ax1.fill_between(t, 0, individual_impact, alpha=0.3, color=color)
        ax1.plot(t, individual_impact, color=color, linewidth=1.5, label=f'Ad {i+1} (t={ad_time})')

        # Mark ad exposure
        ax1.scatter([ad_time], [weight], color=color, s=100, zorder=5, edgecolors='black', linewidth=1)

    ax1.set_ylabel('Individual Ad Effect (Weight)', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', ncol=3, framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.1, which='minor', linestyle=':')
    ax1.minorticks_on()
    ax1.set_ylim(0, 2.2)

    # Bottom panel: Total adstock
    ax2.set_title('Cumulative Ad Stock Over Time', fontweight='bold', pad=10)
    ax2.fill_between(t, 0, adstock, alpha=0.4, color='steelblue', label='Total Ad Stock')
    ax2.plot(t, adstock, color='steelblue', linewidth=2.5)

    # Mark ad exposures
    for ad_time in ad_times:
        ax2.axvline(x=ad_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

    # Add threshold line
    threshold = 1.5
    ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=1.5,
                label=f'Conversion Threshold = {threshold}')

    # Highlight periods above threshold
    above_threshold = adstock >= threshold
    ax2.fill_between(t, threshold, adstock, where=above_threshold,
                     color='green', alpha=0.2, label='High Conversion Probability')

    ax2.set_xlabel('Time (days)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Total Ad Stock $\\sum_k x_{ik}(t)$', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, which='major')
    ax2.grid(True, alpha=0.1, which='minor', linestyle=':')
    ax2.minorticks_on()
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/adstock_accumulation.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/adstock_accumulation.png', bbox_inches='tight')
    plt.close()

# Plot 3: Attribution Example
def plot_attribution_example():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Setup from the numerical example
    ad1_time = 10
    ad2_time = 50
    conversion_time = 60
    beta_A = 0.05
    beta_B = 0.08
    lambda_param = 0.1

    t = np.linspace(0, 80, 800)

    # Calculate adstock for each ad
    adstock_A = np.zeros_like(t)
    adstock_B = np.zeros_like(t)

    mask_A = t >= ad1_time
    adstock_A[mask_A] = exponential_decay(t[mask_A] - ad1_time, lambda_param)

    mask_B = t >= ad2_time
    adstock_B[mask_B] = exponential_decay(t[mask_B] - ad2_time, lambda_param)

    # Calculate causal influence
    influence_A = beta_A * adstock_A
    influence_B = beta_B * adstock_B
    total_influence = influence_A + influence_B

    # Left panel: Ad stock decay
    ax1.set_title('Ad Stock Decay Over Time', fontweight='bold', pad=10)
    ax1.plot(t, adstock_A, label='Ad 1 Stock (Creative A)', color='orange', linewidth=2)
    ax1.plot(t, adstock_B, label='Ad 2 Stock (Creative B)', color='blue', linewidth=2)

    # Mark events
    ax1.scatter([ad1_time], [1], color='orange', s=150, zorder=5,
                edgecolors='black', linewidth=2, label='Ad 1 Shown')
    ax1.scatter([ad2_time], [1], color='blue', s=150, zorder=5,
                edgecolors='black', linewidth=2, label='Ad 2 Shown')
    ax1.axvline(x=conversion_time, color='green', linestyle='--', linewidth=2,
                label='Conversion Event')

    # Show remaining stock at conversion
    conv_idx = np.argmin(np.abs(t - conversion_time))
    ax1.scatter([conversion_time], [adstock_A[conv_idx]], color='orange', s=100,
                zorder=5, edgecolors='black', linewidth=1)
    ax1.scatter([conversion_time], [adstock_B[conv_idx]], color='blue', s=100,
                zorder=5, edgecolors='black', linewidth=1)

    # Annotations
    ax1.annotate(f'Stock = {adstock_A[conv_idx]:.3f}',
                xy=(conversion_time, adstock_A[conv_idx]),
                xytext=(conversion_time + 3, adstock_A[conv_idx] + 0.05),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10)
    ax1.annotate(f'Stock = {adstock_B[conv_idx]:.3f}',
                xy=(conversion_time, adstock_B[conv_idx]),
                xytext=(conversion_time + 3, adstock_B[conv_idx] + 0.05),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10)

    ax1.set_xlabel('Time (days)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Ad Stock Level', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.1, which='minor', linestyle=':')
    ax1.minorticks_on()
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 1.1)

    # Right panel: Attribution pie chart
    ax2.set_title('Attribution at Conversion Time', fontweight='bold', pad=10)

    # Calculate attribution percentages
    attr_A = influence_A[conv_idx] / total_influence[conv_idx] * 100
    attr_B = influence_B[conv_idx] / total_influence[conv_idx] * 100

    # Pie chart
    wedges, texts, autotexts = ax2.pie([attr_A, attr_B],
                                        labels=['Ad 1 (Creative A)', 'Ad 2 (Creative B)'],
                                        colors=['orange', 'blue'],
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=(0, 0.1))

    # Enhance text
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for i, autotext in enumerate(autotexts):
        # Use black text for better readability on colored backgrounds
        autotext.set_color('black')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    # Add details box
    details = (f'Ad 1: t={ad1_time}, β={beta_A}\n'
              f'Ad 2: t={ad2_time}, β={beta_B}\n'
              f'Conversion: t={conversion_time}\n'
              f'Decay: λ={lambda_param}')
    ax2.text(1.25, 0.5, details, transform=ax2.transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/attribution_example.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/attribution_example.png', bbox_inches='tight')
    plt.close()

# Generate all plots
if __name__ == "__main__":
    print("Generating adstock visualization plots...")

    print("1. Creating decay kernels comparison...")
    plot_decay_kernels()

    print("2. Creating adstock accumulation example...")
    plot_adstock_accumulation()

    print("3. Creating attribution example...")
    plot_attribution_example()

    print(f"All plots saved to {output_dir}/")