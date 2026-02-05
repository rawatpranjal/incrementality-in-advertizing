import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Consistent, colorblind-safe palette (Okabe–Ito)
PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermillion': '#D55E00',
    'sky': '#56B4E9',
    'gray': '#595959'
}

# Figure: minimal and horizontal (slightly larger)
fig, ax = plt.subplots(figsize=(12.5, 3.6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Timeline baseline
y0 = 0.45
ax.axhline(y=y0, color=PALETTE['gray'], linewidth=1.2, zorder=1)

# Events: within-session clusters on a compressed time axis (days)
events = [
    # Session 1
    (0.1, 'auction'), (0.35, 'impression'), (0.6, 'click'), (1.1, 'impression'), (1.5, 'purchase'),
    # Session 2
    (5.2, 'auction'), (5.35, 'impression'), (5.6, 'click'), (6.0, 'impression'), (6.4, 'purchase')
]

EVENT_STYLE = {
    'auction':   dict(marker='o', color=PALETTE['blue']),
    'impression':dict(marker='o', color=PALETTE['orange']),
    'click':     dict(marker='s', color=PALETTE['green']),
    'purchase':  dict(marker='D', color=PALETTE['vermillion'])
}

# Session pill boxes (start, end, label)
sessions = [
    (-0.3,  2.0,  'Session 1'),
    ( 4.9,  6.8,  'Session 2')
]

for start, end, lab in sessions:
    rect = FancyBboxPatch((start, y0 - 0.22), end - start, 0.44,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=PALETTE['sky'], alpha=0.22,
                          edgecolor=PALETTE['blue'], linewidth=1.5, zorder=0)
    ax.add_patch(rect)
    ax.text((start + end) / 2, y0 + 0.35, lab, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=PALETTE['blue'])

# Plot events
for x, et in events:
    style = EVENT_STYLE[et]
    ax.scatter(x, y0, s=75, c=style['color'], marker=style['marker'],
               zorder=2, edgecolors='white', linewidth=0.6)

# Gap annotations (> 3 days)
def gap(x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=PALETTE['vermillion'], lw=1.8))
    ax.text((x1 + x2) / 2, y - 0.1, '> 3 days', ha='center', va='top',
            fontsize=9, color=PALETTE['vermillion'], fontweight='bold')

gap(2.0, 4.9, y0 - 0.28)

# Minimal legend, single row
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['blue'],      markersize=7, label='Auction'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['orange'],    markersize=7, label='Impression'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['green'],     markersize=7, label='Click'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=PALETTE['vermillion'],markersize=7, label='Purchase')
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=False, ncol=4)

# Subtle day labels
for d in [0, 2, 4, 6, 8, 10]:
    ax.text(d, y0 - 0.38, f'Day {d}', ha='center', va='top', fontsize=8, color=PALETTE['gray'])

# Limits and cleanliness
ax.set_xlim(-1, 12)
ax.set_ylim(-0.2, 1.0)
ax.axis('off')
plt.tight_layout()

# Save outputs next to script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'session_construction.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'session_construction.png'), dpi=300, bbox_inches='tight')

print(f"Session construction diagram saved to: {output_dir}/session_construction.pdf")
print(f"Session construction diagram saved to: {output_dir}/session_construction.png")
