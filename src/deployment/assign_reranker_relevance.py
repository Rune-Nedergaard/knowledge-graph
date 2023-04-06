"""
This script takes a dict of rephrased questions and their semantic search results, then it does the following:
1. It creates a gpt3.5 prompt template that asks for the relevance of a paragraph to a question and inputs the question and paragraph
2. It extracts the results and creates triplets of (question, paragraph, relevance)
3. It saves these to a dataformat that a BERT model can read and fine tune
"""

### I NEED TO ALSO KEEP TRACK OF THE IDS OF THE QUESTIONS I THINK -- PROBABLY ALSO THE PARAGRAPHS #####

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
    template = ''''Input spørgsmål: {question}
                    
Paragraph 1: {list_of_paragraphs[0]}
                    
Paragraph 2: {list_of_paragraphs[1]}
                    
Paragraph 3: {list_of_paragraphs[2]}
                    
Paragraph 4: {list_of_paragraphs[3]}

Paragraph 5: {list_of_paragraphs[4]}

Paragraph 6: {list_of_paragraphs[5]}

Paragraph 7: {list_of_paragraphs[6]}'''
    REQUEST = template.format(question=question, list_of_paragraphs=list_of_paragraphs)
    #count the number of tokens in the request
  
    while retry < 1:
        try:
            MODEL = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": '''Du vurderer hvor relevante 7 forskellige paragraffer er i forhold til et givent input spørgsmål. Hver paragraf vurderes på en skala fra 0-1, hvor 0 er "slet ikke relevant" og 1 er "meget relevant". Det er sandsynligt at ingen er relevante, så vær hård i din vurdering og giv kun en høj score, hvis det faktisk er relevant. Svar uden yderligere forklaring med et decimaltal per paragraf, således:
                    
                    Paragraf 1: 0.1
                    Pargraf 2: 0.6
                    Paragraf 3: ....'''},

                    {"role": "user", "content": REQUEST}
                ],
                temperature=0,
                max_tokens=100,
            )

            response_text = response['choices'][0]['message']['content']

            with open(os.path.join(output_folder, f'{id}.txt'), 'w', encoding='iso-8859-1', errors='replace') as f:
                f.write(response_text)
                #also write the question and the list of paragraphs to the file
                f.write('\n')
                f.write(question)
                f.write('\n')
                for paragraph in list_of_paragraphs:
                    f.write(paragraph)
                    f.write('\n')
                break# very important to break out of the while loop
        except Exception as e:
            print(e)
            retry += 1
            if retry == 10:
                print(f"Failed to process question {id}")
                failed_files.append(id)
                break

if __name__ == '__main__':

    #Load the semantic search results
    with open('data/semantic_search_results.pkl', 'rb') as f: #remember to change this when running from terminal
        semantic_search_results = pickle.load(f)

    #filenames are the keys of the semantic_search_results dict
    filenames = list(semantic_search_results.keys())

    for id in tqdm(filenames, total=len(filenames)):
        #get the question and the list of paragraphs
        question = semantic_search_results[id][1]
        list_of_paragraphs = semantic_search_results[id][0]
        process_question(question, list_of_paragraphs)