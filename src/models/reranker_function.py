import numpy as np
import os
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from danlp.models import load_bert_base_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_rerank import BertRerank

danish_bert = load_bert_base_model()
tokenizer = danish_bert.tokenizer

# Load the fine-tuned re-ranker model
fine_tuned_model_path = 'models/fine_tuned_model/check7_model.pth'

reranker = BertRerank(model_path=fine_tuned_model_path)



def rerank_paragraphs(question, top_k_paragraphs, reranker):
    # Rerank the paragraphs based on their relevance to the question
    paragraph_scores = []
    
    for paragraph in tqdm(top_k_paragraphs, desc="Reranking paragraphs"):
        similarity = reranker.predict_similarity(question, paragraph)
        paragraph_scores.append((paragraph, similarity))

    # Sort the paragraphs based on their relevance scores
    reranked_paragraphs = sorted(paragraph_scores, key=lambda x: x[1], reverse=True)
    
    return reranked_paragraphs