import os
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import AdamW
from danlp.models import load_bert_base_model
from danlp.models.bert_models import BertBase as DaNLPBertBase
from sklearn.metrics.pairwise import cosine_similarity
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#import the semantic search function
#from src.features.two_towers_fine_tune_seed_optimization import BertCustomBase, TwoTowerSimilarityModel, TwoTowerModel
from src.features.two_towers_fine_tune_multiplenegatives import *
import torch
import faiss
import pickle
import numpy as np

# Load the pre-trained and fine-tuned model
save_directory = 'models/fine_tuned_model_two_tower'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the Danish BERT question model (BertCustomBase)
model_path = 'models/fine_tuned_model_two_tower/model.pt'
two_tower_similarity_model = torch.load(model_path)

pretrained_danish_bert = BertCustomBase(device)

tokenizer = pretrained_danish_bert.tokenizer #not sure it is needed


def embed_question(model, question, tokenizer, device):
    inputs = tokenizer.encode_plus(question, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        question_embedding = model.two_tower_model.forward_question(input_ids, attention_mask)
    return question_embedding.cpu().numpy()



# Load the Faiss index and the mapping dictionary
index_ivfflat = faiss.read_index('index_ivfflat.faiss')
with open('id_mapping_faiss.pickle', 'rb') as f:
    id_mapping = pickle.load(f)

# Load the TwoTowerSimilarityModel
two_tower_model_path = 'models/fine_tuned_model_two_tower/model.pt'
model = torch.load(two_tower_model_path)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Take user input question
user_question = input("Please enter your question: ")

# Embed the user question
question_embedding = embed_question(model, user_question, tokenizer, device)

# Search the 100 nearest paragraphs using the Faiss index
k = 100
distances, indices = index_ivfflat.search(question_embedding, k)

# Display the results
for i, index in enumerate(indices[0]):
    original_id = id_mapping[index]
    print(f"Result {i + 1}:")
    print(f"Original ID: {original_id}")
    print(f"Distance: {distances[0][i]}\n")
