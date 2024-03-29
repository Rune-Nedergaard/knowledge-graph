import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math
from danlp.models import load_bert_base_model
import re
from tqdm import tqdm
import concurrent.futures
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
Filtered_fil = pd.read_pickle('data/raw/Filtered_fil.pkl')

text_files = Filtered_fil['id'].tolist()

# Get the set of files that already exist in the output folder
output_files = {int(os.path.basename(file).split('.')[0]) for file in glob.glob('data/all_paragraphs/paragraphs/*.txt')}

# Find the files that need to be processed
files_to_process = list(set(text_files) - output_files)

print(f"Processing {len(files_to_process)} files")

model = load_bert_base_model()

newline_pattern = re.compile(r'\n')
sentence_split_pattern = re.compile('[?.]')

def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5 * x)))


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum 
        Returns:
            list: list of weighted sums
        """
        p_size = min(p_size, similarities.shape[0])

        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid) 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

def calculate_sentence_embeddings(sentences, model):
    with torch.no_grad():
        sentence_embeddings = [model.embed_text(sentence)[1] for sentence in sentences]
    return np.vstack([embedding.numpy().flatten() for embedding in sentence_embeddings])


def split_into_paragraphs(text, max_tokens=500, model=model):
    text = newline_pattern.sub(' ', text)
    sentences = sentence_split_pattern.split(text)

    # Concatenate short sentences
    start = 0
    end = len(sentences) - 1
    while start <= end:
        if len(sentences[start]) < 30:
            if start == end:
                break
            sentences[start] = sentences[start] + sentences[start + 1]
            sentences.pop(start + 1)
            end -= 1
        else:
            start += 1

    # Concatenate long sentences
    text = ''
    for each in sentences:
        while len(each) > 500:
            split = each[:500]
            text += f'{split}. '
            each = each[500:]
        if len(each) < 30:
            text += f'{each} '
        else:
            text += f'{each}. '
    sentences = text.split('. ')
    text = ''
    for each in sentences:
        if len(each) < 30:
            text += f'{each} '
        else:
            text += f'{each}. '

    # Calculate sentence embeddings
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        embeddings = calculate_sentence_embeddings(sentences, model)

    # Calculate cosine similarities between sentences
    similarities = np.matmul(embeddings, embeddings.T)

    # Find relative minima of activated similarities
    activated_similarities = activate_similarities(similarities, p_size=10)
    minmimas = argrelextrema(activated_similarities, np.less, order=2)

    # Split the text into paragraphs based on the relative minima
    paragraphs = []
    current_paragraph = ''
    for i, sentence in enumerate(sentences):
        current_paragraph += sentence + ' '
        if i in minmimas[0]:
            paragraphs.append(current_paragraph)
            current_paragraph = ''

    return paragraphs



def process_file(file):
    output_file = f'data/all_paragraphs/paragraphs/{file}.txt'
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        #print(f"{output_file} already exists, skipping processing for this file.")
        return
    try:
        with open(f'data/processed/{file}.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        paragraphs = split_into_paragraphs(text, model=model)
        with open(f'data/all_paragraphs/paragraphs/{file}.txt', 'w', encoding='utf-8') as f:
            for paragraph in paragraphs:
                f.write(paragraph + '\n')
    except Exception as e:
        pass
        #print(f"Error processing {file}: {e}")

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file) for file in files_to_process]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()