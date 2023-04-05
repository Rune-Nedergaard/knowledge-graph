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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#import the semantic search function
from src.models.semantic_search_function import get_similar_paragraphs

test = get_similar_paragraphs("Bør vi plante flere trær?", k=4, max_tokens=500, before_percent=0.3)
#print(test)
print(len(test))

#Load the rephrased questions by opening all the files in the rephrased_questions folder and create a dict that maps the file name to the question
rephrased_questions = {}
for file in tqdm(os.listdir('data/rephrased_questions'), total = len(os.listdir('data/rephrased_questions'))):
    filename = file.split('.')[0]
    with open(os.path.join('data/rephrased_questions', file), 'r', encoding='utf-8') as f:
        rephrased_questions[filename] = f.read()

import random
#Create a list of 10 random questions from the rephrased questions
random_questions = random.sample(list(rephrased_questions.keys()), 10)

#Feed each of the random questions to the semantic search function and create a dictionary of the results
semantic_search_results = {}
for q in random_questions:
    semantic_search_results[q] = get_similar_paragraphs(q, k=4, max_tokens=500, before_percent=0.3)

#save the dictionary to a file
with open('data/semantic_search_results.pkl', 'wb') as f:
    pickle.dump(semantic_search_results, f)
