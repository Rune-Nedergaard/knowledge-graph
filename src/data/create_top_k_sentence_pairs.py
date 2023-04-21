import os
import pickle
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.two_towers_fine_tune_multiplenegatives import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load embeddings
embeddings_folder = 'data/embeddings'   
data_folder = 'data'

def cos_sim(a: Tensor, b: Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def load_required_embeddings(embeddings_folder, paragraph_ids):
    embeddings = {}
    for paragraph_id in tqdm(paragraph_ids, desc="Loading embeddings"):
        index = 0
        while True:
            file = f"{paragraph_id}.txt_{index}.npy"
            file_path = os.path.join(embeddings_folder, file)
            if os.path.isfile(file_path):
                embedding = torch.tensor(torch.load(file_path))  # Load the embedding using torch.load
                if paragraph_id not in embeddings:
                    embeddings[paragraph_id] = []

                # Load the paragraph text
                paragraph_file = os.path.join(data_folder, "paragraphs", f"{paragraph_id}.txt")
                try:
                    with open(paragraph_file, "r", encoding="utf-8") as p_file:
                        all_paragraphs = p_file.read().strip().split('\n')
                        if index < len(all_paragraphs):  # Update the condition
                            paragraph_text = all_paragraphs[index]
                        else:
                            print(f"Index out of range for paragraph ID: {paragraph_id}, index: {index}")
                            break
                except:
                    #print(f"Unable to load paragraph file: {paragraph_file}")
                    pass
                embeddings[paragraph_id].append((index, embedding, paragraph_text))  # Append the index, the embedding tensor, and the paragraph text
                #print(f"Loaded embedding: {file_path}")  # Print the loaded embedding file path
                index += 1
            else:
                break
    return embeddings

# Load the TwoTowerSimilarity model
two_tower_model_path = 'models/two_tower_checkpoints_multiplenegatives_v4/model_step_84390_epoch_1.pt'
model = torch.load(two_tower_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load question_to_fil
with open('data/raw/question_to_fil.pkl', 'rb') as f:
    question_to_fil = pickle.load(f)

# Load only the required embeddings for the first 100 questions
paragraph_ids = set()
for _, ids in list(question_to_fil.items()):
    if isinstance(ids, (list, tuple)):
        paragraph_ids.update(ids)
    else:
        paragraph_ids.add(ids)

embeddings = load_required_embeddings(embeddings_folder, paragraph_ids)

k = 4
question_paragraph_pairs_top_k = []
num_top_k_paragraphs = []

for question_id, paragraph_ids in tqdm(list(question_to_fil.items()), desc="Processing questions"):
    if not isinstance(paragraph_ids, (list, tuple)):  # Ensure paragraph_ids is a list or tuple
        paragraph_ids = [paragraph_ids]
    question_file = os.path.join(data_folder, "questions_rephrased", f"{question_id}.txt")

    if not os.path.isfile(question_file):
        print(f"File not found: {question_file}")
        continue

    with open(question_file, "r", encoding='iso-8859-1') as q_file:
        question = q_file.read().strip()

    question_embedding = model.question_model.embed_text(question)
    paragraph_embeddings = []

    for paragraph_id in paragraph_ids:
        if paragraph_id in embeddings:
            for embed in embeddings[paragraph_id]:
                paragraph_embeddings.append(embed)
        else:
            print(f"Embedding not found for paragraph ID: {paragraph_id}")

    num_paragraphs = len(paragraph_embeddings)

    if num_paragraphs < 3:
        num_top_k_paragraphs.append(0)
        continue
    actual_k = min(k, num_paragraphs - 1)



    # Calculate cosine similarity and get top-k indices
    paragraph_embedding_tensors = [embed[1].to(device) for embed in paragraph_embeddings if embed is not None]  # Extract only the embeddings and move them to the device

    if not paragraph_embedding_tensors:
        #print(f"Skipping question {question_id} as no paragraph embeddings were found.")
        num_top_k_paragraphs.append(0)
        continue

    # Calculate cosine similarity and get top-k indices
    similarities = cos_sim(question_embedding, torch.stack(paragraph_embedding_tensors))
    
    # Calculate the actual k value, considering the maximum limit of N-1 paragraphs
    _, top_k_indices = torch.topk(similarities, actual_k)

    # Get the top-k paragraphs
    top_k_paragraphs = [paragraph_embeddings[i][2] for i in top_k_indices.squeeze()]

    # Create question-paragraph pairs
    for top_k_paragraph in top_k_paragraphs:
        question_paragraph_pairs_top_k.append((question, top_k_paragraph))

    num_top_k_paragraphs.append(len(top_k_paragraphs))

with open(os.path.join(data_folder, 'filtered_v3_question_chunk_pairs.pkl'), 'wb') as f:
    pickle.dump(question_paragraph_pairs_top_k, f)


#for i, (question, paragraph) in enumerate(question_paragraph_pairs_top_k):
    #print(f"{i+1}. {question} - {paragraph}")

# Plot the distribution of the number of top-k paragraphs
# 
import matplotlib.pyplot as plt

plt.hist(num_top_k_paragraphs, bins=range(0, k+2), align='left', rwidth=0.8)
plt.xlabel('Number of top-k paragraphs')
plt.ylabel('Frequency')
plt.title('Distribution of the number of top-k paragraphs for each question')
plt.xticks(range(0, k+1))
plt.show()