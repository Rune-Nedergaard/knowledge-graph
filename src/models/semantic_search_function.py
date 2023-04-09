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
import hashlib


#from danlp.models import load_bert_base_model



def get_similar_paragraphs(query, k=4, max_tokens=500, before_percent=0.3, approximate=True, embedding_matrix=None, ids=None):#last three are for exact search
    # Load the fine-tuned BERT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model_path = 'models/fine_tuned_model.pth'
    model_path = 'models/fine_tuned_model/check7_model.pth' #this is the new way of loading after I changed the saving process
    model = BertEmbed(model_path=model_path)
    model.model.to(device)

    # Function to create embeddings for the query (question)
    def create_query_embeddings(query):
        sentence_embedding = model.embed_text(query)
        return sentence_embedding

    # Create embeddings for the query
    query_embeddings = create_query_embeddings(query).cpu()
    query_embeddings = np.reshape(query_embeddings, (1, -1))

    if approximate:
        # Load the saved index
        index_ivfflat = faiss.read_index('index_ivfflat.faiss')

        # Perform semantic search
        D, I = index_ivfflat.search(query_embeddings, k)

        # Load the id_mapping_faiss.pickle
        with open('id_mapping_faiss.pickle', 'rb') as f:
            id_mapping_faiss = pickle.load(f)

        # Flip the keys and values in the dictionary
        id_mapping_faiss = {v: k for k, v in id_mapping_faiss.items()}

        # Converting the indices to the actual file names
        I = [id_mapping_faiss[i] for i in I[0]]
    else:
        print("Using the slow exact search")

        embedding_matrix = embedding_matrix
        ids = ids
        # Compute cosine similarity between query and all embeddings
        D = np.dot(embedding_matrix, query_embeddings.T)
        I = np.argsort(D, axis=0)[::-1][:k, 0].tolist()

        # Convert indices to the actual file names
        I = [ids[i] for i in I]

    similar_paragraphs = []
    # Open up the file and get the context paragraphs for each similar item
    for i in I:
        # Getting the basename, before the _ and the index
        basename = i.split('_')[0]
        # Getting the index
        index = int(i.split('_')[1])
        # Opening the file
        with open(os.path.join('data/paragraphs', basename), 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
            similar_paragraphs.append(paragraphs[index])
            # Get the context paragraphs
            #context_paragraphs = get_context_paragraphs(paragraphs, index, max_tokens, before_percent)
            #similar_paragraphs.append(context_paragraphs)

    return similar_paragraphs

"""
def get_context_paragraphs(paragraphs, index, max_tokens=450, before_percent=0.3):
    # Convert paragraphs to raw text and print first 500 characters

    
    #DECIDED AGAINST USING THIS FUNCTION, DUE TO MANY REASONS
    
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

if __name__ == '__main__':
    test = get_similar_paragraphs('Hvad skal vi gøre med den stigende ældrebyrde?', k=1000)
    print(test)