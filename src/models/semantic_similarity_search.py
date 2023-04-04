import faiss
import numpy as np
import h5py
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_embed import BertEmbed
import torch
from tqdm import tqdm
import pickle



# Load the saved index
index_ivfflat = faiss.read_index('index_ivfflat.faiss')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

"""
# THIS NEEDS TO BE CHANGED TO THIS, BUT WORKING WITH PRETRAINED FOR NOW UNTIL FINE TUNED SAVING IS FIXED
# Load the fine-tuned BERT model
model = BertEmbed()
model.model.to(device)
"""

from danlp.models import load_bert_base_model
model = load_bert_base_model()



# Function to create embeddings for the query (question)
def create_query_embeddings(query):
    # The danlp bert just takes raw text and tokenizes within embed_text, not sure about mine

    _, sentence_embedding, _ =model.embed_text(query)

    return sentence_embedding



# Perform semantic search
query = "Bør vi plante flere træeer?"
query_embeddings = create_query_embeddings(query)
query_embeddings = np.reshape(query_embeddings, (1, -1))

k = 5  # Number of results to retrieve

D, I = index_ivfflat.search(query_embeddings, k)

#load the id_mapping_faiss.pickle

with open('id_mapping_faiss.pickle', 'rb') as f:
    id_mapping_faiss = pickle.load(f)

#flip the keys and values in the dictionary
id_mapping_faiss = {v: k for k, v in id_mapping_faiss.items()}

#Converting the indices to the actual file names
I = [id_mapping_faiss[i] for i in I[0]]

#open up the file and print the paragraph
for i in I:
    #getting the basename, befor the _ and the index
    basename = i.split('_')[0]
    #getting the index
    index = i.split('_')[1]
    #opening the file
    with open(os.path.join('data/paragraphs', basename), 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')
        print(paragraphs[int(index)])
        print('\n')
# Print the indices of the top k paragraphs
