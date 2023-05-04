import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.utils import resample

def read_scores_from_folder(folder, score_prefix):
    files = glob.glob(os.path.join(folder, '*.txt'))
    scores = []

    for file in files:
        with open(file, 'r', encoding='iso-8859-1', errors='replace') as f:
            content = f.read()
            scores.extend([float(x) for x in re.findall(fr'{score_prefix} \d: (\d\.\d+)', content)])

    return scores

def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = [np.mean(resample(data)) for _ in range(n_bootstrap)]
    lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

folder_1 = 'data/relevance_scores'
folder_2 = 'data/relevance_scores_no_reranker'
folder_3 = 'data/relevance_scores_qa_pairs'
folder_4 = 'data/relevance_scores_danlp_qa_pairs'

folders = [folder_1, folder_2, folder_3, folder_4]
group_names = ['Reranker', 'No Reranker', 'Multilingual QA pairs', 'DaNLP QA pairs']
score_prefixes = ['Tekststykke', 'Tekststykke', 'Par', 'Par']

data = []

for folder, group_name, score_prefix in zip(folders, group_names, score_prefixes):
    scores = read_scores_from_folder(folder, score_prefix)
    print(f"Number of scores read from {group_name}: {len(scores)}")  # Add this line to print the number of scores
    if len(scores) > 0:  # Add this condition to avoid calculating the mean of an empty list
        ci_lower, ci_upper = bootstrap_confidence_interval(scores)
        data.append({
            'Group': group_name,
            'Mean': np.mean(scores),
            'Lower CI': ci_lower,
            'Upper CI': ci_upper
        })


data_df = pd.DataFrame(data)

fig, ax = plt.subplots()
sns.barplot(x='Group', y='Mean', data=data_df, ax=ax)
ax.errorbar(data_df['Group'], data_df['Mean'], yerr=[data_df['Mean'] - data_df['Lower CI'], data_df['Upper CI'] - data_df['Mean']], fmt='none', c='black', capsize=5)

plt.title('Comparison of Relevance Scores')
plt.xlabel('Group')
plt.ylabel('Relevance Score')
plt.show()
