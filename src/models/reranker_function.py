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



def rerank_paragraphs(question, top_k_paragraphs, reranker, batch_size=100):
    # Rerank the paragraphs based on their relevance to the question
    paragraph_scores = []

    num_paragraphs = len(top_k_paragraphs)
    num_batches = (num_paragraphs + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Reranking paragraphs"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_paragraphs)
        batch_paragraphs = top_k_paragraphs[start_idx:end_idx]

        similarities = [reranker.predict_similarity(question, paragraph) for paragraph in batch_paragraphs]
        paragraph_scores.extend(zip(batch_paragraphs, similarities))

    # Sort the paragraphs based on their relevance scores
    reranked_paragraphs = sorted(paragraph_scores, key=lambda x: x[1], reverse=True)

    # Print the first 5 reranked paragraphs
    for idx, (paragraph, score) in enumerate(reranked_paragraphs[:5]):
        print(f"Paragraph {idx + 1}: {paragraph[:30]}\nRelevance Score: {score}\n")
    
    return reranked_paragraphs