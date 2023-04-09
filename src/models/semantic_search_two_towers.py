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
from src.features.two_towers_fine_tune import BertCustomBase, TwoTowerSimilarityModel, TwoTowerModel

# Load the pre-trained and fine-tuned model
save_directory = 'models/fine_tuned_model_two_tower'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the Danish BERT question model (BertCustomBase)
bert_model_question = BertCustomBase(device)
bert_model_paragraph = BertCustomBase(device)

# Initialize the TwoTowerModel
two_tower_model = TwoTowerModel(bert_model_question, bert_model_paragraph).to(device)

#Initialize the TwoTowerSimilarityModel
two_tower_similarity_model = TwoTowerSimilarityModel(two_tower_model).to(device)

#Load the state dict
two_tower_similarity_model.load_state_dict(torch.load(os.path.join(save_directory, 'model.pt')))
two_tower_similarity_model.to(device) #probably not needed


tokenizer = bert_model_question.tokenizer



# Define a function to encode text using the model
def encode_input(text, bert_model):
    _, _, tokenized_text = bert_model.embed_text(text)
    input_ids = bert_model.tokenizer.encode(tokenized_text, return_tensors='pt', is_split_into_words=True).to(bert_model.device)
    attention_mask = input_ids != bert_model.tokenizer.pad_token_id
    return input_ids, attention_mask


def predict_similarity(question, paragraph):
    question_input_ids, question_attention_mask = encode_input(question, bert_model_question)
    paragraph_input_ids, paragraph_attention_mask = encode_input(paragraph, bert_model_paragraph)

    similarity_score = two_tower_similarity_model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)

    return similarity_score.item()

