#!/usr/bin/env python3
"""
Script to export the comparison table from the article writing example as an image.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a DataFrame with the comparison data
data = {
    'Framework': ['autogen', 'semantic_kernel', 'langchain', 'crewai'],
    'Time (s)': [0.80, 88.09, 109.81, 94.30],
    'Tokens': [8423, 7721, 8270, 974],
    'Cost ($)': [0.0531, 0.0486, 0.0554, 0.0094]
}

df = pd.DataFrame(data)

# Set up the figure and style
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Plot execution time
sns.barplot(x='Framework', y='Time (s)', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('Execution Time (seconds)')
axes[0].set_ylabel('Seconds')

# Plot token usage
sns.barplot(x='Framework', y='Tokens', data=df, ax=axes[1], palette='viridis')
axes[1].set_title('Token Usage')
axes[1].set_ylabel('Number of Tokens')

# Plot cost
sns.barplot(x='Framework', y='Cost ($)', data=df, ax=axes[2], palette='viridis')
axes[2].set_title('Cost (USD)')
axes[2].set_ylabel('USD')

# Adjust layout and add title
plt.tight_layout()
fig.suptitle('Multi-Agent Assessment Framework Comparison', fontsize=16, y=1.05)

# Save the figure
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'framework_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Comparison chart saved to {output_path}")

# Also create a table image
plt.figure(figsize=(10, 4))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table = plt.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center',
    colColours=['#f2f2f2']*len(df.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

table_path = os.path.join(output_dir, 'framework_comparison_table.png')
plt.savefig(table_path, dpi=300, bbox_inches='tight')

print(f"Comparison table saved to {table_path}") 