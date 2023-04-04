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

def get_context_paragraphs(paragraphs, index, max_tokens=500, before_percent=0.3):
    #convert paragraphs to raw text and print first 500 characters
    raw_text = ' '.join(paragraphs)
    print(f"The input starts like this {raw_text[:100]}")
    current_paragraph = paragraphs[index]
    current_length = len(current_paragraph.split())
    remaining_tokens = max_tokens - current_length

    # If the entire document is shorter than max_tokens, return the entire document
    total_tokens = sum([len(p.split()) for p in paragraphs])
    if total_tokens <= max_tokens:
       output = ''.join(paragraphs)
       print("OUTPUT STARTS LIKE THIS:", output[0:50])
       print(f"There are {len(output.split())} tokens in the output.")
       print("Type of output:", type(output))
       print('\n')
       print(('_'*20))
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
    
    #recalculate after_tokens by adding the reamining before tokens that have not been used
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
    print("OUTPUT STARTS LIKE THIS:", output[0:50])
    print(f"There are {len(output.split())} tokens in the output.")
    print("Type of output:", type(output))
    print('\n')
    print(('_'*20))
    return output
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

k = 6  # Number of results to retrieve

D, I = index_ivfflat.search(query_embeddings, k)

#load the id_mapping_faiss.pickle

with open('id_mapping_faiss.pickle', 'rb') as f:
    id_mapping_faiss = pickle.load(f)

#flip the keys and values in the dictionary
id_mapping_faiss = {v: k for k, v in id_mapping_faiss.items()}

#Converting the indices to the actual file names
I = [id_mapping_faiss[i] for i in I[0]]

saved = []
#open up the file and print the paragraph
for i in I:
    #getting the basename, befor the _ and the index
    basename = i.split('_')[0]
    #getting the index
    index = int(i.split('_')[1])
    #opening the file
    with open(os.path.join('data/paragraphs', basename), 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n')
        #get the context paragraphs
        print(f"The name of the file is {basename} and the index is {index}")
        paragraphs = get_context_paragraphs(paragraphs, index)

        #print(f"The number of tokens in the context paragraphs is {sum([len(p.split()) for p in paragraphs])}")
 
