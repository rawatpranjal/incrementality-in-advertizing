import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.5

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.7, 'Panel Construction from Continuous-Time Event Stream',
        ha='center', va='top', fontsize=13, fontweight='bold')

# ============================================================================
# PART 1: TIMELINE WITH EVENTS AND SAMPLING
# ============================================================================

timeline_y = 8.5
timeline_start = 1.5
timeline_end = 12.5

# Draw main timeline
ax.plot([timeline_start, timeline_end], [timeline_y, timeline_y],
        'k-', linewidth=3, alpha=0.7)
ax.text(timeline_start - 0.3, timeline_y, 't = 0', ha='right', va='center', fontsize=10, fontweight='bold')
ax.text(timeline_end + 0.3, timeline_y, 't = T', ha='left', va='center', fontsize=10, fontweight='bold')

# Add time markers
for i in range(0, 31, 5):
    x_pos = timeline_start + (timeline_end - timeline_start) * i / 30
    ax.plot([x_pos, x_pos], [timeline_y - 0.1, timeline_y + 0.1], 'k-', linewidth=1.5, alpha=0.5)
    ax.text(x_pos, timeline_y - 0.3, f'Day {i}', ha='center', va='top', fontsize=7, alpha=0.7)

# Define conversion events (actual purchases)
conversions = [
    {'day': 5, 'user': 'u_42', 'vendor': 'v_7', 'time': '10:23'},
    {'day': 12, 'user': 'u_91', 'vendor': 'v_12', 'time': '08:15'},
    {'day': 18, 'user': 'u_27', 'vendor': 'v_5', 'time': '14:50'},
    {'day': 25, 'user': 'u_63', 'vendor': 'v_9', 'time': '16:42'}
]

# Define random negative samples
negatives = [
    {'day': 8, 'user': 'u_18', 'vendor': 'v_3', 'time': '14:42'},
    {'day': 19, 'user': 'u_55', 'vendor': 'v_8', 'time': '16:30'}
]

# Colors for sample types
color_positive = '#2ca02c'  # green
color_negative = '#ff7f0e'  # orange
color_double = '#9467bd'    # purple

# Plot conversion events on timeline
conversion_positions = []
for conv in conversions:
    x_pos = timeline_start + (timeline_end - timeline_start) * conv['day'] / 30
    conversion_positions.append(x_pos)
    # Draw star for conversion
    ax.scatter([x_pos], [timeline_y], s=250, marker='*', c='red',
               edgecolors='darkred', linewidth=1.5, zorder=10)
    # Label above
    ax.text(x_pos, timeline_y + 0.5, f"Purchase\n{conv['user']}",
            ha='center', va='bottom', fontsize=7, color='darkred')

# Plot random negative sample points
negative_positions = []
for neg in negatives:
    x_pos = timeline_start + (timeline_end - timeline_start) * neg['day'] / 30
    negative_positions.append(x_pos)
    # Draw circle for random sample
    ax.scatter([x_pos], [timeline_y], s=120, marker='o',
               facecolors='none', edgecolors=color_negative, linewidth=2, zorder=9)

# Add sampling annotation boxes
sample_y = 9.3

# P+ box
ax.text(2.5, sample_y, r'$\mathcal{P}^+$: All conversions',
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_positive, alpha=0.2, edgecolor=color_positive, linewidth=2))

# P- box
ax.text(7, sample_y, r'$\mathcal{P}^-$: Random samples',
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_negative, alpha=0.2, edgecolor=color_negative, linewidth=2))

# P0 box
ax.text(11.5, sample_y, r'$\mathcal{P}^0$: Duplicates of $\mathcal{P}^+$',
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_double, alpha=0.2, edgecolor=color_double, linewidth=2))

# ============================================================================
# PART 2: PANEL TABLE WITH EXAMPLE ROWS
# ============================================================================

table_y_start = 6.8
row_height = 0.5

# Table header
header_y = table_y_start
ax.text(7, header_y + 0.3, 'Final Panel: Example Rows',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Column positions
cols = {
    'row': 1.5,
    'user': 2.5,
    'vendor': 3.7,
    'timestamp': 5.3,
    'y': 7.3,
    'weight': 8.5,
    'type': 10.0,
    'adstock': 11.8
}

# Draw table header
header_y_line = table_y_start - 0.1
for col_name, x_pos in cols.items():
    ax.text(x_pos, header_y_line, col_name.upper(),
            ha='center', va='center', fontsize=8, fontweight='bold')

# Draw header line
ax.plot([1.0, 13.0], [header_y_line - 0.25, header_y_line - 0.25],
        'k-', linewidth=2)

# Create table rows (alternating P+, P0, P-, ...)
rows_data = []
row_id = 1

# Add rows for each conversion (P+ and P0)
for i, conv in enumerate(conversions[:2]):  # Show first 2 conversions
    # P+ row
    rows_data.append({
        'row': row_id,
        'user': conv['user'],
        'vendor': conv['vendor'],
        'timestamp': f"Day {conv['day']} {conv['time']}",
        'y': '1',
        'weight': '+1.0',
        'type': r'$\mathcal{P}^+$',
        'adstock': f'{np.random.uniform(0.6, 0.9):.2f}',
        'color': color_positive,
        'timeline_pos': conversion_positions[i]
    })
    row_id += 1

    # P0 row (duplicate)
    rows_data.append({
        'row': row_id,
        'user': conv['user'],
        'vendor': conv['vendor'],
        'timestamp': f"Day {conv['day']} {conv['time']}",
        'y': '0',
        'weight': '−1.0',
        'type': r'$\mathcal{P}^0$',
        'adstock': f'{rows_data[-1]["adstock"]}',  # Same as P+
        'color': color_double,
        'timeline_pos': conversion_positions[i]
    })
    row_id += 1

# Add P- rows
for i, neg in enumerate(negatives):
    rows_data.append({
        'row': row_id,
        'user': neg['user'],
        'vendor': neg['vendor'],
        'timestamp': f"Day {neg['day']} {neg['time']}",
        'y': '0',
        'weight': '+342',
        'type': r'$\mathcal{P}^-$',
        'adstock': f'{np.random.uniform(0.0, 0.3):.2f}',
        'color': color_negative,
        'timeline_pos': negative_positions[i]
    })
    row_id += 1

# Draw table rows
current_y = header_y_line - 0.5
for i, row_data in enumerate(rows_data):
    # Alternate background
    if i % 2 == 0:
        rect = mpatches.Rectangle((1.0, current_y - row_height/2 + 0.05), 12.0, row_height - 0.1,
                                   facecolor='lightgray', alpha=0.15, zorder=-1)
        ax.add_patch(rect)

    # Draw row values
    for col_name, x_pos in cols.items():
        value = row_data.get(col_name, '')
        if col_name == 'type':
            # Colored box for sample type
            ax.text(x_pos, current_y, value, ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=row_data['color'],
                            alpha=0.3, edgecolor=row_data['color'], linewidth=1))
        else:
            ax.text(x_pos, current_y, str(value), ha='center', va='center', fontsize=8)

    # Draw connecting arrow from timeline to table row
    arrow_start_y = timeline_y - 0.15
    arrow_end_y = current_y + 0.2
    arrow = FancyArrowPatch(
        (row_data['timeline_pos'], arrow_start_y),
        (cols['timestamp'], arrow_end_y),
        arrowstyle='->', mutation_scale=12, linewidth=1.5,
        color=row_data['color'], alpha=0.4, linestyle='--', zorder=1
    )
    ax.add_patch(arrow)

    current_y -= row_height

# Draw table border
ax.plot([1.0, 13.0], [current_y + 0.15, current_y + 0.15], 'k-', linewidth=2)

# ============================================================================
# KEY INSIGHTS BOX
# ============================================================================

insight_y = 0.8
insight_box = FancyBboxPatch((1.3, insight_y), 11.4, 1.2,
                             boxstyle="round,pad=0.1",
                             edgecolor='#1f77b4', facecolor='#1f77b4',
                             alpha=0.08, linewidth=2)
ax.add_patch(insight_box)

ax.text(7, insight_y + 0.85, 'Key Observations:',
        ha='center', va='top', fontsize=10, fontweight='bold', color='#1f77b4')

insights = [
    r'• $\mathcal{P}^+$ and $\mathcal{P}^0$ have identical covariates (timestamp, user, vendor, ad stock) but different $y$ and weight',
    r'• $\mathcal{P}^-$ samples random times; large weight ($w^- = |\mathcal{A}| \cdot T / n^-$) represents entire non-conversion space',
    r'• Negative weight in $\mathcal{P}^0$ corrects for over-representation of converters in regression moments'
]

text_y = insight_y + 0.55
for insight in insights:
    ax.text(7, text_y, insight, ha='center', va='top', fontsize=8, color='#1f77b4')
    text_y -= 0.25

plt.tight_layout()
plt.savefig('panel_construction_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Timeline-based diagram saved as panel_construction_diagram.png")