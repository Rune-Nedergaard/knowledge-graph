import os
import h5py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_embed import BertEmbed
import torch
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the fine-tuned BERT model

model = BertEmbed()

model.model.to(device)

# Read paragraphs from files in the data/paragraphs folder and generate embeddings
embeddings = {}
file_dir = 'data/paragraphs'

for filename in tqdm(os.listdir(file_dir), desc='Embedding paragraphs'):
    if filename.endswith('.txt'):
        with open(os.path.join(file_dir, filename), 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
            file_embeddings = []
            for index, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    embedding = model.embed_text(paragraph)
                    file_embeddings.append((index, embedding))
            embeddings[filename] = file_embeddings

# Save embeddings using h5py
saved_with_h5py = False

try:
    with h5py.File('embeddings.h5', 'w') as f:
        for filename, file_embeddings in embeddings.items():
            group = f.create_group(filename)
            for index, embedding in file_embeddings:
                group.create_dataset(str(index), data=embedding.cpu().numpy())
    print('Embeddings saved to embeddings.h5')
    saved_with_h5py = True
except Exception as e:
    print("Failed to save embeddings with h5py. Error:", e)

if not saved_with_h5py:
    try:
        with open('embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved to embeddings.pickle")
    except Exception as e:
        print("Failed to save embeddings with pickle. Error:", e)