"""
This script takes a dict of rephrased questions and their semantic search results, then it does the following:
1. It creates a gpt3.5 prompt template that asks for the relevance of a paragraph to a question and inputs the question and paragraph
2. It extracts the results and creates triplets of (question, paragraph, relevance)
3. It saves these to a dataformat that a BERT model can read and fine tune
"""


import openai
from api_secrets import API_KEY
import os
import re
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import multiprocessing
import pickle

openai.api_key = API_KEY
failed_files = []

output_folder = 'data/reranker_triplets'

def process_question(question, list_of_paragraphs):
    retry = 0
    while retry < 4:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Du omskriver spørgsmål til mere generelle spørgsmål uden mange detaljer. Den omskrevne tekst skal minde om noget en normal person ville bruge til at søge efter information online og skal ikke være adresseret til nogen bestemt."},
                    {"role": "user", "content": text},
                ],
                temperature=0.5,
                max_tokens=100,
            )

            response_text = response['choices'][0]['message']['content']

            with open(os.path.join(output_folder, f'{id}.txt'), 'w') as f:
                f.write(response_text)
            break
        except Exception as e:
            print(e)
            retry += 1
            if retry == 10:
                print(f"Failed to process question {id}")
                failed_files.append(id)
                break

if __name__ == '__main__':

    #Load the semantic search results
    with open('data/semantic_search_results.pkl', 'rb') as f:
        semantic_search_results = pickle.load(f)

    #Load the question_to_fil dict
    with open('data/raw/question_to_fil.pkl', 'rb') as f:
        question_to_fil = pickle.load(f)

    questions = list(semantic_search_results.keys())

    #get the fil for each question
    fils = [question_to_fil[question] for question in questions]


    print(semantic_search_results)

