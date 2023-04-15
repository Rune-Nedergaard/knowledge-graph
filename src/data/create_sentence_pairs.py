import os
import pickle
from tqdm import tqdm

with open('data/raw/question_to_fil_filtered.pkl', 'rb') as f:
    question_to_fil = pickle.load(f)

data_folder = 'data'

def get_question_chunk_pairs(data_folder, question_to_fil):
    question_chunk_pairs = []

    for question_id, paragraph_ids in tqdm(question_to_fil.items(), desc="Processing questions"):
        question_file = os.path.join(data_folder, "questions_rephrased", f"{question_id}.txt")

        if not os.path.isfile(question_file):
            print(f"File not found: {question_file}")
            continue

        with open(question_file, "r", encoding='iso-8859-1') as q_file:
            question = q_file.read().strip()

        if not isinstance(paragraph_ids, (list, tuple)):  # Ensure paragraph_ids is a list or tuple
            paragraph_ids = [paragraph_ids]

        for paragraph_id in paragraph_ids:
            paragraph_file = os.path.join(data_folder, "all_paragraphs_large_removed/paragraphs", f"{paragraph_id}.txt")

            if not os.path.isfile(paragraph_file):
                print(f"File not found: {paragraph_file}")
                continue

            with open(paragraph_file, "r", encoding='iso-8859-1') as p_file:
                paragraphs = p_file.read().split("\n")

            for paragraph in paragraphs:
                if paragraph:  # Check if paragraph is not empty
                    question_chunk_pairs.append((question, paragraph))

    return question_chunk_pairs

data_folder = 'data'

question_chunk_pairs = get_question_chunk_pairs(data_folder, question_to_fil)
#save as pickle
with open(os.path.join(data_folder, 'big_question_chunk_pairs.pkl'), 'wb') as f:
    pickle.dump(question_chunk_pairs, f)
