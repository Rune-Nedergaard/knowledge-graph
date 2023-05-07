import seaborn as sns
import matplotlib.pyplot as plt

# Set the seaborn whitegrid theme
sns.set_theme(style='whitegrid')

# Create a dataset with the desired values and labels
data = {
    'Model': ['GPT-4', 'QA search'],
    'Score (0-50)': [47, 44]
}

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='Score (0-50)', data=data, ax=ax)

ax.set_ylim(0, 50)
ax.set_yticks(range(0, 51, 10))
ax.set_title('SDU Journalist Knowledge Test')
ax.set_ylabel('Score')

# Show the plot
plt.show()
