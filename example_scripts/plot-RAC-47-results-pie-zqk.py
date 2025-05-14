import matplotlib.pyplot as plt
import numpy as np

# Data for the custom evaluation method
labels_custom = ['Good', 'Sparse', 'Hallucination', 'Not Used']
sizes_custom = [16, 4, 23, 4]

# Define new color palettes consistent with previous changes
color_good = '#6495ED'        # Cornflower Blue (consistent with 'Successful')
color_sparse = '#FFD700'      # Gold (for 'Sparse')
color_hallucination = '#FFA07A' # Light Salmon (consistent with 'Failed')
color_not_used = '#B0C4DE'    # Light Steel Blue (consistent with 'Not Used')

colors_custom = [color_good, color_sparse, color_hallucination, color_not_used]

# Set global font sizes for PPT-friendly output
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['figure.titlesize'] = 24

# Create the pie chart with transparent background
fig_custom, ax_custom = plt.subplots(figsize=(8, 8), facecolor='none')
ax_custom.pie(sizes_custom, labels=labels_custom, colors=colors_custom,
              autopct='%1.1f%%', shadow=False, startangle=140, pctdistance=0.7)

# Ensure the pie chart is a circle
ax_custom.axis('equal')

# Add a title
plt.title('ProToken Performance on RAC-47 (Geometric & Continuity)')

# Save the figure with transparent background
plt.savefig('rac47_zqk_evaluation.png', dpi=300, bbox_inches='tight', transparent=True)

# Show the plot (optional)
# plt.show()

print("Custom evaluation pie chart 'rac47_custom_evaluation.png' saved with transparent background.")
