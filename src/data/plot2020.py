import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# Load the data
data = pd.read_csv('data/OF2020results.csv')

# Print the first five rows of the data
print(data.head())

# Extract the scores as a list
scores = data['Score'].tolist()

# Set the style for the plot
sns.set_style("whitegrid")
#set palette
sns.set_palette("muted")

# Create a histogram using seaborn
plt.figure(figsize=(10,6))
sns.histplot(scores, bins=15, kde=False, stat='count', alpha=0.8, edgecolor='black', linewidth=1)

# Add title, axis labels, and adjust layout
plt.title('Distribution of Scores (Weighted)')
plt.xlabel('Score (0-100)')
plt.xlim(0, 100)
plt.ylabel('Count')
plt.tight_layout()

# Add vertical lines with legends
plt.axvline(55, linestyle='-', color='mediumaquamarine', label='GPT-3.5',linewidth=2)
plt.axvline(87, linestyle='-', color='magenta', label='GPT-4',linewidth=2)
plt.axvline(82, linestyle='-', color='orange', label='QA, t=0.7', linewidth=2.5)
plt.axvline(84, linestyle='-', color='mediumblue', label='QA, t=0.5',linewidth=2)
plt.axvline(82, linestyle='--', color='green', label='QA, t=0.5 constrained')
plt.axvline(50, linestyle='-', color='black', label='Passing score',linewidth=1)

# Add a custom legend
legend_elements = [plt.Line2D([0], [0], linestyle='-', color='mediumaquamarine', label='GPT-3.5'),
                   plt.Line2D([0], [0], linestyle='-', color='magenta', label='GPT-4'),
                   plt.Line2D([0], [0], linestyle='-', color='orange', label='QA, t=0.7'),
                   plt.Line2D([0], [0], linestyle='-', color='mediumblue', label='QA, t=0.5'),
                   plt.Line2D([0], [0], linestyle='--', color='green', label='QA, t=0.5 constrained'),
                   plt.Line2D([0], [0], linestyle='-', color='black', label='Passing score')]
plt.legend(handles=legend_elements, loc='upper left')

# Add text annotation for the sample size
plt.text(0.05, -0.1, f'n = {len(scores)}', transform=plt.gca().transAxes,
         va='center', ha='left')

# Adjust the plot layout to make space for the annotation and legend
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.8)

# Add a box around the plot
plt.box(True)
plt.xticks(np.arange(0, 101, 10))

# Show the plot
plt.show()
