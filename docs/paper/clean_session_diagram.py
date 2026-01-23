import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from datetime import datetime, timedelta

# Set style for clean academic look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

# Create figure with specific size for paper
fig, ax = plt.subplots(figsize=(10, 4))

# Define colors - muted academic palette
colors = {
    'auction': '#4472C4',      # Blue
    'impression': '#ED7D31',    # Orange
    'click': '#70AD47',        # Green
    'purchase': '#C55A11'       # Dark orange
}

# Timeline setup
base_time = datetime(2025, 3, 1, 8, 0)

# Define events for cleaner visualization
events = [
    # Macro-Session 1 (Days 0-1.5)
    (base_time, 'auction'),
    (base_time + timedelta(minutes=5), 'impression'),
    (base_time + timedelta(minutes=10), 'click'),
    (base_time + timedelta(hours=3), 'impression'),
    (base_time + timedelta(days=1), 'auction'),
    (base_time + timedelta(days=1, hours=2), 'click'),
    (base_time + timedelta(days=1.3), 'purchase'),

    # Gap > 3 days -> New macro-session
    (base_time + timedelta(days=5), 'auction'),
    (base_time + timedelta(days=5, minutes=30), 'impression'),
    (base_time + timedelta(days=5.5), 'click'),
    (base_time + timedelta(days=6), 'impression'),

    # Gap > 3 days -> New macro-session
    (base_time + timedelta(days=10), 'auction'),
    (base_time + timedelta(days=10, minutes=15), 'impression'),
    (base_time + timedelta(days=10, minutes=30), 'click'),
    (base_time + timedelta(days=10.2), 'purchase'),
]

# Y position for timeline
y_base = 0.5

# Plot events as vertical lines with markers
for time, event_type in events:
    x_pos = (time - base_time).total_seconds() / 86400  # Convert to days

    # Draw vertical line
    ax.plot([x_pos, x_pos], [y_base - 0.02, y_base + 0.15],
            color='gray', linewidth=0.5, alpha=0.5, zorder=1)

    # Add event marker
    marker_style = {'auction': 'o', 'impression': '^', 'click': 's', 'purchase': 'D'}
    ax.scatter(x_pos, y_base + 0.15,
               s=80,
               c=colors[event_type],
               marker=marker_style[event_type],
               zorder=3,
               edgecolors='white',
               linewidth=1.5,
               alpha=0.9)

# Draw macro-session boundaries as shaded regions
session_regions = [
    (0, 1.5, 'Session 1'),     # Days 0-1.5
    (5, 6.2, 'Session 2'),      # Days 5-6.2
    (10, 10.3, 'Session 3'),    # Days 10-10.3
]

for start, end, label in session_regions:
    # Draw shaded region
    rect = Rectangle((start, y_base - 0.25), end - start, 0.5,
                     facecolor='#E8F2FD',
                     edgecolor='#4472C4',
                     linewidth=1.5,
                     alpha=0.6,
                     zorder=0)
    ax.add_patch(rect)

    # Add session label below
    ax.text((start + end) / 2, y_base - 0.15, label,
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            color='#2E4057')

# Draw the timeline axis
ax.axhline(y=y_base, color='black', linewidth=1.5, zorder=2)

# Add gap annotations with cleaner arrows
# Gap 1
ax.annotate('', xy=(1.5, y_base), xytext=(5, y_base),
            arrowprops=dict(arrowstyle='<->',
                          color='#C55A11',
                          lw=1.5,
                          shrinkA=0, shrinkB=0))
ax.text(3.25, y_base + 0.03, '3.5 days',
        ha='center', va='bottom',
        fontsize=9, color='#C55A11')

# Gap 2
ax.annotate('', xy=(6.2, y_base), xytext=(10, y_base),
            arrowprops=dict(arrowstyle='<->',
                          color='#C55A11',
                          lw=1.5,
                          shrinkA=0, shrinkB=0))
ax.text(8.1, y_base + 0.03, '3.8 days',
        ha='center', va='bottom',
        fontsize=9, color='#C55A11')

# Create legend with better positioning
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colors['auction'], markersize=8,
               label='Auction', markeredgecolor='white'),
    plt.Line2D([0], [0], marker='^', color='w',
               markerfacecolor=colors['impression'], markersize=8,
               label='Impression', markeredgecolor='white'),
    plt.Line2D([0], [0], marker='s', color='w',
               markerfacecolor=colors['click'], markersize=8,
               label='Click', markeredgecolor='white'),
    plt.Line2D([0], [0], marker='D', color='w',
               markerfacecolor=colors['purchase'], markersize=8,
               label='Purchase', markeredgecolor='white'),
]

# Place legend in upper left
legend = ax.legend(handles=legend_elements,
                   loc='upper left',
                   frameon=True,
                   fancybox=False,
                   shadow=False,
                   framealpha=1,
                   edgecolor='gray',
                   fontsize=9,
                   ncol=4,
                   columnspacing=1)

# Style settings
ax.set_xlim(-0.5, 11)
ax.set_ylim(0, 0.8)
ax.set_xlabel('Days from First Event', fontsize=11)

# Remove y-axis
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Style x-axis
ax.set_xticks(range(0, 12, 2))
ax.spines['bottom'].set_position(('data', 0))

# Title
ax.set_title('Macro-Session Construction with 3-Day Inactivity Threshold',
             fontsize=12, fontweight='bold', pad=15)

# Add definition text box in upper right
definition_text = (
    "Definition: A macro-session ends when\n"
    "user inactivity exceeds 3 days.\n"
    "Each session captures an extended\n"
    "shopping episode."
)
ax.text(0.98, 0.95, definition_text,
        transform=ax.transAxes,
        fontsize=9,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3',
                 facecolor='white',
                 edgecolor='gray',
                 alpha=0.9))

# Adjust layout
plt.tight_layout()

# Save figures
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.pdf',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("Clean session diagram saved as session_construction.pdf and session_construction.png")