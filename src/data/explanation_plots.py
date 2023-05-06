import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

# Define the categories and weighted sums
categories = ['Agree-Correct', 'Agree-Incorrect', 'Disagree-Correct', 'Disagree-Incorrect']
explanation_weighted_sums = [64, 12, 58, 16]
no_explanation_weighted_sums = [64, 9, 60, 17]

# Create a grouped bar plot
bar_width = 0.35
x_pos = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(10, 6))  # Set the plot size
rects1 = ax.bar(x_pos - bar_width/2, explanation_weighted_sums, bar_width, label='With Explanation')
rects2 = ax.bar(x_pos + bar_width/2, no_explanation_weighted_sums, bar_width, label='No Explanation')

ax.set_ylabel('Weighted Sums')
ax.set_title('Effect of Explanations on Respondent Agreement/Disagreement')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
