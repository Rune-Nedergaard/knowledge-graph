import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import sem

data = pd.read_csv('data/explanation.csv')


data = data.iloc[:44]  # Keep only rows 2 to 45 (Python uses 0-based indexing)

# Display the first few rows of the data
print(data.head())

print(data.columns)

# Assign the correct column names
column_A = "Unnamed: 0"
column_B = "Christian"
column_C = "Kasper"
column_D = "Nicolai"
column_E = "Facit"
column_G = "0.5cons"
column_H = "C svar"
column_I = "K svar"
column_J = "N svar"

# Extract the data from the DataFrame into lists
question_numbers = data[column_A].tolist()
correct_answers = data[column_E].tolist()
model_answers = data[column_G].tolist()
R1_answers = data[column_H].tolist()
R2_answers = data[column_I].tolist()
R3_answers = data[column_J].tolist()

explanation_status_R1 = data[column_B].tolist()
explanation_status_R2 = data[column_C].tolist()
explanation_status_R3 = data[column_D].tolist()

# Initialize lists for each category
data_agree_correct_explanation = []
data_agree_correct_no_explanation = []
data_agree_incorrect_explanation = []
data_agree_incorrect_no_explanation = []
data_disagree_correct_explanation = []
data_disagree_correct_no_explanation = []
data_disagree_incorrect_explanation = []
data_disagree_incorrect_no_explanation = []

# Iterate through the data and categorize each answer
for q_num, correct, model, R1, R2, R3, exp_R1, exp_R2, exp_R3 in zip(
    question_numbers, correct_answers, model_answers, 
    R1_answers, R2_answers, R3_answers,
    explanation_status_R1, explanation_status_R2, explanation_status_R3):

    # Check the categories for Respondent 1
    if R1 == model:
        if model == correct:
            if exp_R1 == 1:
                data_agree_correct_explanation.append(q_num)
            else:
                data_agree_correct_no_explanation.append(q_num)
        else:
            if exp_R1 == 1:
                data_agree_incorrect_explanation.append(q_num)
            else:
                data_agree_incorrect_no_explanation.append(q_num)
    else:
        if model == correct:
            if exp_R1 == 1:
                data_disagree_correct_explanation.append(q_num)
            else:
                data_disagree_correct_no_explanation.append(q_num)
        else:
            if exp_R1 == 1:
                data_disagree_incorrect_explanation.append(q_num)
            else:
                data_disagree_incorrect_no_explanation.append(q_num)


    if R2 == model:
        if model == correct:
            if exp_R2 == 1:
                data_agree_correct_explanation.append(q_num)
            else:
                data_agree_correct_no_explanation.append(q_num)
        else:
            if exp_R2 == 1:
                data_agree_incorrect_explanation.append(q_num)
            else:
                data_agree_incorrect_no_explanation.append(q_num)
    else:
        if model == correct:
            if exp_R2 == 1:
                data_disagree_correct_explanation.append(q_num)
            else:
                data_disagree_correct_no_explanation.append(q_num)
        else:
            if exp_R2 == 1:
                data_disagree_incorrect_explanation.append(q_num)
            else:
                data_disagree_incorrect_no_explanation.append(q_num)

    if R3 == model:
        if model == correct:
            if exp_R3 == 1:
                data_agree_correct_explanation.append(q_num)
            else:
                data_agree_correct_no_explanation.append(q_num)
        else:
            if exp_R3 == 1:
                data_agree_incorrect_explanation.append(q_num)
            else:
                data_agree_incorrect_no_explanation.append(q_num)
    else:
        if model == correct:
            if exp_R3 == 1:
                data_disagree_correct_explanation.append(q_num)
            else:
                data_disagree_correct_no_explanation.append(q_num)
        else:
            if exp_R3 == 1:
                data_disagree_incorrect_explanation.append(q_num)
            else:
                data_disagree_incorrect_no_explanation.append(q_num)


# Calculate the counts for each category
count_ACE = len(data_agree_correct_explanation)
count_AIE = len(data_agree_incorrect_explanation)
count_ACN = len(data_agree_correct_no_explanation)
count_AIN = len(data_agree_incorrect_no_explanation)
count_DCE = len(data_disagree_correct_explanation)
count_DIE = len(data_disagree_incorrect_explanation)
count_DCN = len(data_disagree_correct_no_explanation)
count_DCE2 = len(data_disagree_incorrect_no_explanation)

# Print the counts
print(f"A-C-E: {count_ACE}")
print(f"A-I-E: {count_AIE}")
print(f"A-C-N: {count_ACN}")
print(f"A-I-N: {count_AIN}")
print(f"D-C-E: {count_DCE}")
print(f"D-I-E: {count_DIE}")
print(f"D-C-N: {count_DCN}")
print(f"D-C-E: {count_DCE2}")



sns.set_style("whitegrid")
#set  palette to muted
#sns.set_palette("muted")
#change colors
#sns.set_palette("Set2")

# Define the categories and unweighted sums
categories = ['Agree-Correct', 'Agree-Incorrect', 'Disagree-Correct', 'Disagree-Incorrect']
explanation_data = [data_agree_correct_explanation, data_agree_incorrect_explanation, data_disagree_correct_explanation, data_disagree_incorrect_explanation]
no_explanation_data = [data_agree_correct_no_explanation, data_agree_incorrect_no_explanation, data_disagree_correct_no_explanation, data_disagree_incorrect_no_explanation]

# Create a grouped bar plot with confidence intervals and square error bars
bar_width = 0.35
x_pos = np.arange(len(categories))
cap_size = 5  # Set the size of the error bar caps

fig, ax = plt.subplots(figsize=(10, 6))  # Set the plot size
rects1 = ax.bar(x_pos - bar_width/2, [len(i) for i in explanation_data], bar_width, label='With Explanation', yerr=[max(1.96*np.std(i)/np.sqrt(len(i)), 0) for i in explanation_data], error_kw={'capsize': cap_size})
rects2 = ax.bar(x_pos + bar_width/2, [len(i) for i in no_explanation_data], bar_width, label='No Explanation', yerr=[max(1.96*np.std(i)/np.sqrt(len(i)), 0) for i in no_explanation_data], error_kw={'capsize': cap_size})

ax.set_ylabel('Unweighted Count')
ax.set_title('Effect of Explanations comparing Respondent-Model Agreement/Disagreement by Model Correctness')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.set_ylim([0, 35])
ax.legend()

plt.tight_layout()
plt.show()
