import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Minimal color palette
COLORS = {
    'session_fill': '#DBEAFE',
    'session_border': '#2563EB',
    'event': '#1E40AF',
    'timeline': '#374151',
    'gap_arrow': '#DC2626',
    'text': '#111827',
    'subtle': '#6B7280'
}

# Set up minimal figure
fig, ax = plt.subplots(figsize=(12, 3))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Timeline y-position
y_line = 0.5

# Draw clean timeline
ax.axhline(y=y_line, color=COLORS['timeline'], linewidth=1.5, zorder=1)

# Events: (day, type)
events = [
    # Session 1
    (0.2, 'search'), (0.5, 'search'), (0.8, 'click'), (1.2, 'click'), (1.5, 'purchase'),
    # Session 2
    (5.1, 'search'), (5.4, 'click'), (5.8, 'search'), (6.2, 'click'), (6.5, 'click'),
    # Session 3
    (10.0, 'search'), (10.2, 'click'), (10.5, 'purchase'),
]

EVENT_STYLES = {
    'search': {'color': '#6B7280', 'marker': 'o'},      # gray circle
    'click': {'color': '#F59E0B', 'marker': 'o'},       # amber circle
    'purchase': {'color': '#10B981', 'marker': 's'},    # green square
}

# Session boundaries: (start, end, label)
sessions = [
    (-0.2, 1.8, 'Session 1'),
    (4.8, 6.8, 'Session 2'),
    (9.7, 10.9, 'Session 3'),
]

# Draw pill-shaped sessions
for start, end, label in sessions:
    rect = FancyBboxPatch(
        (start, y_line - 0.25), end - start, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['session_fill'],
        edgecolor=COLORS['session_border'],
        linewidth=2,
        zorder=2
    )
    ax.add_patch(rect)
    ax.text((start + end) / 2, y_line + 0.38, label,
            ha='center', va='bottom', fontsize=10,
            fontweight='bold', color=COLORS['session_border'])

# Draw event dots
for x, etype in events:
    style = EVENT_STYLES[etype]
    ax.scatter(x, y_line, s=70, c=style['color'], marker=style['marker'],
               zorder=3, edgecolors='white', linewidth=0.5)

# Minimal legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6B7280', markersize=7, label='Search'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F59E0B', markersize=7, label='Click'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#10B981', markersize=7, label='Purchase'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9,
          edgecolor='none', ncol=3, columnspacing=1)

# Gap annotations
ax.annotate('', xy=(1.8, y_line - 0.4), xytext=(4.8, y_line - 0.4),
            arrowprops=dict(arrowstyle='<->', color=COLORS['gap_arrow'], lw=2))
ax.text(3.3, y_line - 0.52, '>3 days', ha='center', va='top', fontsize=9,
        color=COLORS['gap_arrow'], fontweight='bold')

ax.annotate('', xy=(6.8, y_line - 0.4), xytext=(9.7, y_line - 0.4),
            arrowprops=dict(arrowstyle='<->', color=COLORS['gap_arrow'], lw=2))
ax.text(8.25, y_line - 0.52, '>3 days', ha='center', va='top', fontsize=9,
        color=COLORS['gap_arrow'], fontweight='bold')

# Day markers
for day in [0, 2, 4, 6, 8, 10]:
    ax.text(day, y_line - 0.7, f'Day {day}', ha='center', va='top',
            fontsize=8, color=COLORS['subtle'])

# Arrow at end
ax.annotate('', xy=(11.5, y_line), xytext=(11, y_line),
            arrowprops=dict(arrowstyle='->', color=COLORS['timeline'], lw=1.5))

# Clean styling
ax.set_xlim(-1, 12)
ax.set_ylim(-0.3, 1.0)
ax.axis('off')

plt.tight_layout()

# Save
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.pdf',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/pranjal/Code/marketplace-incrementality/latex/session_construction.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("Macro-session diagram saved as session_construction.pdf and session_construction.png")