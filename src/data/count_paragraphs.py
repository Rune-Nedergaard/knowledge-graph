import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
data_folder = 'data'

# Load question_to_fil
with open('data/raw/question_to_fil.pkl', 'rb') as f:
    question_to_fil = pickle.load(f)

num_paragraphs_per_question = []

for question_id, paragraph_ids in tqdm(question_to_fil.items(), desc="Processing questions"):
    if not isinstance(paragraph_ids, (list, tuple)):  # Ensure paragraph_ids is a list or tuple
        paragraph_ids = [paragraph_ids]

    total_paragraphs = 0

    for paragraph_id in paragraph_ids:
        paragraph_file = os.path.join(data_folder, "paragraphs", f"{paragraph_id}.txt")

        if not os.path.isfile(paragraph_file):
            continue

        with open(paragraph_file, "r", encoding="utf-8") as p_file:
            all_paragraphs = p_file.read().strip().split('\n')
            total_paragraphs += len(all_paragraphs)

    num_paragraphs_per_question.append(total_paragraphs)

# Plot the distribution of the number of paragraphs per question
counted_paragraphs = Counter(num_paragraphs_per_question)
plt.bar(counted_paragraphs.keys(), counted_paragraphs.values())
plt.xlabel('Number of paragraphs')
plt.xlim(0, 50)
plt.ylim(0, 10000)
plt.ylabel('Frequency')
plt.title('Distribution of the number of paragraphs for each question')
plt.show()
