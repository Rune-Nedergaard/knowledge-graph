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
import faiss
danish_bert = load_bert_base_model()
tokenizer = danish_bert.tokenizer

# Load the fine-tuned re-ranker model
#fine_tuned_model_path = 'models/fine_tuned_model/check_highbatch2_model.pth'

#reranker = BertRerank(model_path=fine_tuned_model_path)
index_ivfflat = faiss.read_index('tt_index_ivfflat.faiss')
with open('tt_id_mapping_faiss.pickle', 'rb') as f:
    id_mapping = pickle.load(f)
#id_mapping = {value: key for key, value in id_mapping.items()}

def get_context_paragraphs(filename, index, max_tokens=350, before_percent=0.3):
    # Convert paragraphs to raw text and print first 500 characters
    with open(os.path.join('data/all_paragraphs/paragraphs', filename), 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')

    if index >= len(paragraphs):
        print("Failed to get context paragraphs for index", index, "in file", filename)
        print("Returning empty string")
        return ''
        #raise ValueError(f'Index {index} is out of bounds for {filename}')

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


"""
def get_filename_from_index(index_tuple):
    combined_index = '_'.join([str(i) for i in index_tuple])
    original_id = id_mapping[combined_index]
    return original_id
    #return original_id.split('_')[0] + '.txt'
"""



def rerank_paragraphs(question, file_indices, top_k_paragraphs, reranker):
    paragraph_scores = []

    for file_tuple, index, paragraph in tqdm(zip(file_indices, range(len(top_k_paragraphs)), top_k_paragraphs), desc="Reranking paragraphs"):
        filename = file_tuple[0]
        paragrah_index = file_tuple[1]
        #if index != file_tuple[1]:
        #    print("Index mismatch")
        #filename is the stuff before .txt
        #filename = txtname.split('.')[0]


        similarity = reranker.predict_similarity(question, paragraph)
        paragraph_scores.append((paragrah_index, filename, paragraph, similarity))

    reranked_paragraphs = sorted(paragraph_scores, key=lambda x: x[3], reverse=True)
    reranked_paragraphs = reranked_paragraphs[:10]
    reranked_paragraphs = [(get_context_paragraphs(filename, paragraph_index), score, filename, paragraph_index) for paragraph_index, filename, _, score in reranked_paragraphs]

    return reranked_paragraphs
