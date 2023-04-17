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
from src.features.two_towers_fine_tune_multiplenegatives import TwoTowerSimilarityModel, BertCustomBase
import faiss


def get_similar_paragraphs(user_question, k=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_danish_bert = BertCustomBase(device)

    # Load the Faiss index and the mapping dictionary
    index_ivfflat = faiss.read_index('tt_index_ivfflat.faiss')
    with open('tt_id_mapping_faiss.pickle', 'rb') as f:
        id_mapping = pickle.load(f)

    id_mapping = {value: key for key, value in id_mapping.items()}

    similar_paragraphs = []
    # Load the TwoTowerSimilarityModel
    model_path = 'models/two_tower_checkpoints_multiplenegatives_v4/model_step_84390_epoch_1.pt'
    model = torch.load(model_path, map_location=device)

    with torch.no_grad():
        question_embedding = model.question_model.embed_text(user_question).cpu().numpy()
        question_embedding = question_embedding.reshape(1, -1)

    # Search the nearest paragraphs using the Faiss index
    distances, indices = index_ivfflat.search(question_embedding, k)

    # Retrieve the paragraphs
    for index in indices[0]:
        if index in id_mapping:
            original_id = id_mapping[index]
            basename = original_id.split('.txt_')[0] + '.txt'
            paragraph_index = int(original_id.split('.txt_')[1])

            with open(os.path.join('data/all_paragraphs/paragraphs', basename), 'r', encoding='utf-8') as f:
                paragraphs = f.read().split('\n')
                similar_paragraphs.append(paragraphs[paragraph_index])

    return similar_paragraphs


if __name__ == '__main__':



    user_question = input("Please enter your question: ")

    # Get the list of similar paragraphs
    similar_paragraphs = get_similar_paragraphs(user_question)

    # Print the similar paragraphs
    for i, paragraph in enumerate(similar_paragraphs):
        print(f"Result {i + 1}:")
        print(paragraph)
        print("\n")
