import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from datetime import datetime, timedelta

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors for different event types
colors = {
    'auction': '#1f77b4',
    'impression': '#ff7f0e',
    'click': '#2ca02c',
    'purchase': '#d62728'
}

# Create timeline for a sample user
base_time = datetime(2025, 3, 1, 8, 0)
events = [
    # Session 1
    (base_time, 'auction'),
    (base_time + timedelta(minutes=1), 'impression'),
    (base_time + timedelta(minutes=2), 'click'),
    (base_time + timedelta(minutes=15), 'auction'),
    (base_time + timedelta(minutes=16), 'impression'),
    (base_time + timedelta(hours=1), 'auction'),
    (base_time + timedelta(hours=1, minutes=1), 'impression'),
    (base_time + timedelta(hours=1, minutes=2), 'click'),
    (base_time + timedelta(hours=1, minutes=30), 'purchase'),

    # Gap > 2 hours -> New session within same journey
    (base_time + timedelta(hours=4), 'auction'),
    (base_time + timedelta(hours=4, minutes=1), 'impression'),
    (base_time + timedelta(hours=4, minutes=30), 'auction'),
    (base_time + timedelta(hours=4, minutes=31), 'impression'),
    (base_time + timedelta(hours=4, minutes=32), 'click'),

    # Still within 48-hour journey window
    (base_time + timedelta(hours=20), 'auction'),
    (base_time + timedelta(hours=20, minutes=1), 'impression'),

    # New journey (> 48 hours from journey start)
    (base_time + timedelta(hours=50), 'auction'),
    (base_time + timedelta(hours=50, minutes=1), 'impression'),
    (base_time + timedelta(hours=50, minutes=2), 'click'),
]

# Plot events on timeline
y_position = 0.5
for time, event_type in events:
    x_pos = (time - base_time).total_seconds() / 3600  # Convert to hours
    ax.scatter(x_pos, y_position, s=150, c=colors[event_type],
               marker='o' if event_type != 'purchase' else 's',
               zorder=3, edgecolors='black', linewidth=1.5)

    # Add event labels
    ax.text(x_pos, y_position + 0.05, event_type[0].upper(),
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Draw session boundaries
session_boundaries = [
    (0, 1.5, 'Session 1'),  # First session
    (4, 5, 'Session 2'),     # Second session (after 2-hour gap)
    (20, 20.5, 'Session 3'), # Third session
]

for start, end, label in session_boundaries:
    # Draw session box
    rect = FancyBboxPatch((start, 0.35), end - start, 0.3,
                          boxstyle="round,pad=0.01",
                          facecolor='lightblue', alpha=0.3,
                          edgecolor='blue', linewidth=2)
    ax.add_patch(rect)

    # Add session label
    ax.text((start + end) / 2, 0.25, label, ha='center', va='top',
            fontsize=10, fontweight='bold')

# Draw journey boundaries
journey_boundaries = [
    (0, 48, 'Journey 1: USER123_2025030108'),
    (50, 52, 'Journey 2: USER123_2025030310')
]

for start, end, label in journey_boundaries:
    # Draw journey box
    rect = FancyBboxPatch((start, 0.1), end - start, 0.8,
                          boxstyle="round,pad=0.02",
                          facecolor='lightgreen', alpha=0.15,
                          edgecolor='green', linewidth=2.5, linestyle='--')
    ax.add_patch(rect)

    # Add journey label
    ax.text((start + end) / 2, 0.95, label, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='darkgreen')

# Add annotations for gaps
ax.annotate('', xy=(1.5, 0.5), xytext=(4, 0.5),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(2.75, 0.55, '2.5 hour gap\n→ New Session', ha='center', va='bottom',
        fontsize=9, color='red', fontweight='bold')

ax.annotate('', xy=(48, 0.5), xytext=(50, 0.5),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text(49, 0.55, '> 48 hours\n→ New Journey', ha='center', va='bottom',
        fontsize=9, color='purple', fontweight='bold')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['auction'],
               markersize=10, label='Auction', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['impression'],
               markersize=10, label='Impression', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['click'],
               markersize=10, label='Click', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['purchase'],
               markersize=10, label='Purchase', markeredgecolor='black'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add parameter box
param_text = """Key Parameters:
• Journey Window: 48 hours
• Session Gap: 2 hours
• Session ID: UserID_YYYYMMDDHH
• Events aggregated within sessions"""

ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
        fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Formatting
ax.set_xlim(-1, 55)
ax.set_ylim(0, 1.2)
ax.set_xlabel('Time (hours from first event)', fontsize=12)
ax.set_ylabel('')
ax.set_yticks([])
ax.set_title('Shopping Session Construction: From Raw Events to Analytical Units',
             fontsize=14, fontweight='bold', pad=20)

# Add grid for time reference
ax.grid(True, axis='x', alpha=0.3, linestyle=':')
ax.set_xticks(range(0, 56, 4))

# Add explanation text at bottom
explanation = ("Sessions group user events with < 2-hour gaps. Journeys span 48-hour windows. "
               "Each session becomes one observation for analysis.")
ax.text(0.5, -0.1, explanation, transform=ax.transAxes,
        fontsize=11, ha='center', va='top', style='italic', wrap=True)

plt.tight_layout()

# Save the figure
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.png',
            dpi=300, bbox_inches='tight')

print("Session construction diagram saved as session_construction.pdf and session_construction.png")