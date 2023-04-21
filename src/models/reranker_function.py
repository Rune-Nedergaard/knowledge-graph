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



def get_context_paragraphs(paragraphs, index, max_tokens=450, before_percent=0.3):
    # Convert paragraphs to raw text and print first 500 characters
    
    current_paragraph = paragraphs[index]
    current_length = len(current_paragraph.split())
    remaining_tokens = max_tokens - current_length

    # If the entire document is shorter than max_tokens, return the entire document
    total_tokens = sum([len(p.split()) for p in paragraphs])
    if total_tokens <= max_tokens:
       output = ''.join(paragraphs)
       return output

    # Calculate how many tokens should be assigned to the paragraphs before and after the current paragraph
    before_tokens = int(remaining_tokens * before_percent)
    after_tokens = remaining_tokens - before_tokens

    # Initialize the start and end indices of the window
    start_index = index
    end_index = index

    # Keep adding paragraphs to the window until the desired number of tokens is reached
    while before_tokens > 0 and start_index > 0:
        start_index -= 1
        paragraph_tokens = len(paragraphs[start_index].split())
        if before_tokens >= paragraph_tokens:
            before_tokens -= paragraph_tokens
        else:
            start_index += 1
            break

    # Recalculate after_tokens by adding the remaining before tokens that have not been used
    after_tokens = after_tokens + before_tokens

    while after_tokens > 0 and end_index < len(paragraphs) - 1:
        end_index += 1
        paragraph_tokens = len(paragraphs[end_index].split())
        if after_tokens >= paragraph_tokens:
            after_tokens -= paragraph_tokens
        else:
            end_index -= 1
            break

    # Return the window of paragraphs
    output = ' '.join(paragraphs[start_index:end_index + 1])
    return output



def rerank_paragraphs(question, indices, top_k_paragraphs, reranker):
    # Rerank the paragraphs based on their relevance to the question
    paragraph_scores = []
    
    for index, paragraph in tqdm(zip(indices, top_k_paragraphs), desc="Reranking paragraphs"):
        similarity = reranker.predict_similarity(question, paragraph)
        paragraph_scores.append((index, paragraph, similarity))

    # Sort the paragraphs based on their relevance scores
    reranked_paragraphs = sorted(paragraph_scores, key=lambda x: x[2], reverse=True)
    
    #top 10 paragraphs
    reranked_paragraphs = reranked_paragraphs[:10]
    
    #using the get context paragraphs function to get the context paragraphs for the top 10 paragraphs
    reranked_paragraphs = [get_context_paragraphs(paragraphs, index) for index, paragraphs, similarity in reranked_paragraphs]
    
    return reranked_paragraphs
