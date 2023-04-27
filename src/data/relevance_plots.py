import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.utils import resample

def read_scores_from_folder(folder):
    files = glob.glob(os.path.join(folder, '*.txt'))
    scores = []

    for file in files:
        with open(file, 'r', encoding='iso-8859-1', errors='replace') as f:
            content = f.read()
            scores.extend([float(x) for x in re.findall(r'Tekststykke \d: (\d\.\d+)', content)])

    return scores

def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = [np.mean(resample(data)) for _ in range(n_bootstrap)]
    lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

folder_1 = 'data/relevance_scores'
folder_2 = 'data/relevance_scores_no_reranker'

scores_1 = read_scores_from_folder(folder_1)
scores_2 = read_scores_from_folder(folder_2)

ci_lower_1, ci_upper_1 = bootstrap_confidence_interval(scores_1)
ci_lower_2, ci_upper_2 = bootstrap_confidence_interval(scores_2)

data = pd.DataFrame({
    'Group': ['Reranker', 'No Reranker'],
    'Mean': [np.mean(scores_1), np.mean(scores_2)],
    'Lower CI': [ci_lower_1, ci_lower_2],
    'Upper CI': [ci_upper_1, ci_upper_2]
})

fig, ax = plt.subplots()
sns.barplot(x='Group', y='Mean', data=data, ax=ax)
ax.errorbar(data['Group'], data['Mean'], yerr=[data['Mean'] - data['Lower CI'], data['Upper CI'] - data['Mean']], fmt='none', c='black', capsize=5)

plt.title('Comparison of Relevance Scores')
plt.xlabel('Group')
plt.ylabel('Relevance Score')
plt.show()
