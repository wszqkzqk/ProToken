#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes for PPT-friendly output
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['legend.fontsize'] = 24

# Define a helper function to create individual pie charts
def create_pie_chart(sizes, labels, colors, title_text, filename):
    """
    Creates and saves a single pie chart.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=False, startangle=160, pctdistance=0.7)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title_text)
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    # plt.show() # Optional: display plot
    print(f"Saved pie chart: {filename}")

# --- Data and Plotting for Individual Charts ---

# Define new color palettes
color_successful = '#6495ED'  # Cornflower Blue
color_failed = '#FFA07A'      # Light Salmon
color_not_used = '#B0C4DE'    # Light Steel Blue (or keep #9E9E9E Grey)

# 1. ProToken using UFConf criteria
labels_protoken_ufconf = ['Successful', 'Failed', 'Not Used']
sizes_protoken_ufconf = [42, 1, 4]
colors_protoken_ufconf = [color_successful, color_failed, color_not_used]
create_pie_chart(sizes_protoken_ufconf, labels_protoken_ufconf, colors_protoken_ufconf,
                 'ProToken Performance on RAC-47 (UFConf Criteria)',
                 'protoken_ufconf_evaluation.png')

# 2. UFConf (from UFConf paper)
labels_ufconf_paper = ['Successful', 'Failed']
sizes_ufconf_paper = [19, 28] # Total 47
colors_ufconf_paper = [color_successful, color_failed]
create_pie_chart(sizes_ufconf_paper, labels_ufconf_paper, colors_ufconf_paper,
                 'UFConf Performance on RAC-47 (Reported in UFConf Paper)',
                 'ufconf_original_evaluation.png')

# 3. AlphaFold3 (from UFConf paper)
labels_af3_paper = ['Successful', 'Failed']
sizes_af3_paper = [11, 36] # Total 47
colors_af3_paper = [color_successful, color_failed]
create_pie_chart(sizes_af3_paper, labels_af3_paper, colors_af3_paper,
                 'AlphaFold3 Performance on RAC-47 (Reported in UFConf Paper)',
                 'alphafold3_ufconf_evaluation.png')

# 4. AlphaFlow (from UFConf paper)
labels_aflow_paper = ['Successful', 'Failed']
sizes_aflow_paper = [18, 29] # Total 47
colors_aflow_paper = [color_successful, color_failed]
create_pie_chart(sizes_aflow_paper, labels_aflow_paper, colors_aflow_paper,
                 'AlphaFlow Performance on RAC-47 (Reported in UFConf Paper)',
                 'alphaflow_ufconf_evaluation.png')


# --- Plotting for Combined PPT Chart ---
fig_ppt, axs_ppt = plt.subplots(2, 2, figsize=(15, 14)) # Adjusted figsize for better text fit
fig_ppt.suptitle('Comparison of Methods on RAC-47 (UFConf Evaluation Criteria)', fontsize=20)

# Plot 1: ProToken
axs_ppt[0, 0].pie(sizes_protoken_ufconf, labels=labels_protoken_ufconf, colors=colors_protoken_ufconf,
                  autopct='%1.1f%%', shadow=False, startangle=140, pctdistance=0.80)
axs_ppt[0, 0].set_title('ProToken (This Study)')
axs_ppt[0, 0].axis('equal')

# Plot 2: UFConf (Paper)
axs_ppt[0, 1].pie(sizes_ufconf_paper, labels=labels_ufconf_paper, colors=colors_ufconf_paper,
                  autopct='%1.1f%%', shadow=False, startangle=140, pctdistance=0.80)
axs_ppt[0, 1].set_title('UFConf (Paper)')
axs_ppt[0, 1].axis('equal')

# Plot 3: AlphaFold3 (Paper)
axs_ppt[1, 0].pie(sizes_af3_paper, labels=labels_af3_paper, colors=colors_af3_paper,
                  autopct='%1.1f%%', shadow=False, startangle=140, pctdistance=0.80)
axs_ppt[1, 0].set_title('AlphaFold3 (Paper)')
axs_ppt[1, 0].axis('equal')

# Plot 4: AlphaFlow (Paper)
axs_ppt[1, 1].pie(sizes_aflow_paper, labels=labels_aflow_paper, colors=colors_aflow_paper,
                  autopct='%1.1f%%', shadow=False, startangle=140, pctdistance=0.80)
axs_ppt[1, 1].set_title('AlphaFlow (Paper)')
axs_ppt[1, 1].axis('equal')

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.savefig('ufconf_comparison_evaluation_ppt.png', dpi=300, transparent=False)
print("Saved combined PPT pie chart: 'ufconf_comparison_evaluation_ppt.png'")
