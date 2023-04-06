"""
This script takes a rephrased question, then it does the following:
1. It performs semantic serach using the question and the embedded paragraph corpus
2. It then takes the top 4 paragraphs and creates a pair of (question, paragraph), making sure that at most 2 paragraphs are from the file belonging to the question
3 It then takes a random paragraph from the corpus and creates a pair of (question, paragraph)
4. It does this for 2000 questions
5. It then saves the pairs to a file
"""

import os
import pickle
from tqdm import tqdm
import sys
import h5py
import hashlib
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#import the semantic search function
from src.models.semantic_search_function import get_similar_paragraphs


#Load the rephrased questions by opening all the files in the rephrased_questions folder and create a dict that maps the file name to the question
rephrased_questions = {}
for file in tqdm(os.listdir('data/questions_rephrased'), total = len(os.listdir('data/questions_rephrased'))):
    filename = file.split('.')[0]
    #using iso-8859-1 encoding because some of the files have non-utf-8 characters
    with open(os.path.join('data/questions_rephrased', file), 'r', encoding='iso-8859-1') as f:
        rephrased_questions[filename] = f.read()

import random
#Create a list of 10 random questions from the rephrased questions
random_questions_ids = random.sample(list(rephrased_questions.keys()), 10)
#these are the ids of the questions

"""
experimenting with exact search to see if that improves things
"""

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get the total number of embeddings and their dimension
        num_embeddings = sum([len(group) for group in f.values()])
        dimension = f[next(iter(f))]['0'].shape[0]

        # Initialize the embedding matrix and ID list
        embedding_matrix = np.empty((num_embeddings, dimension), dtype=np.float32)
        ids = []

        # Use a set to keep track of unique embeddings
        unique_embeddings = set()

        # Fill the embedding matrix and ID list
        i = 0
        for filename, group in tqdm(f.items(), total = len(f), desc='Loading embeddings'):
            for index, dataset in group.items():
                # Compute a hash of the embedding as a string
                embedding_hash = hashlib.sha256(np.ascontiguousarray(dataset[:]).tobytes()).hexdigest()
                # Add the embedding and ID to the lists if it has not already been seen
                if embedding_hash not in unique_embeddings:
                    unique_embeddings.add(embedding_hash)
                    ids.append(f"{filename}_{index}")
                    embedding_matrix[i] = dataset[:]
                    i += 1

        # Truncate the embedding matrix and ID list to remove duplicates
        embedding_matrix = embedding_matrix[:i]
        ids = ids[:i]

    return embedding_matrix, ids

embeddings_matrix, ids = load_embeddings('embeddings.h5')

#Feed each of the random questions to the semantic search function and create a dictionary of the results
semantic_search_results = {}
for id in random_questions_ids:
    q = rephrased_questions[id]
    #saving as tuple with both a list of the paragraphs and the i
    semantic_search_results[id] = (get_similar_paragraphs(q, k=4, max_tokens=450, before_percent=0.3, approximate=False, embedding_matrix=embeddings_matrix, ids=ids), q)

#save the dictionary to a file
with open('data/semantic_search_results.pkl', 'wb') as f:
    pickle.dump(semantic_search_results, f)
