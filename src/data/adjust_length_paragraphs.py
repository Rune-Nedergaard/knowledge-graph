import os
import torch
import numpy as np
from danlp.models import load_bert_base_model
from scipy.spatial.distance import cosine
from collections import defaultdict
from tqdm import tqdm
import glob


# Load the Danish BERT model and tokenizer
danish_bert = load_bert_base_model()
tokenizer = danish_bert.tokenizer
bert_model = danish_bert.model


def read_and_adjust_paragraphs_from_files(file_paths, min_tokens, max_tokens):
    adjusted_paragraphs_by_file = {}

    for file_path in tqdm(file_paths, desc="Reading and adjusting paragraphs", total=len(file_paths)):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                paragraphs = content.split('\n')
        except:
            print(f"Could not read file {file_path} as utf-8, trying iso-8859-1")
            try:
                with open(file_path, 'r', encoding='iso-8859-1') as file:
                    content = file.read()
                    paragraphs = content.split('\n')
            except:
                print(f"Could not read file {file_path} as iso-8859-1 either, skipping")
                continue

        adjusted_paragraphs = adjust_paragraph_length(paragraphs, min_tokens, max_tokens)
        adjusted_paragraphs_by_file[file_path] = adjusted_paragraphs
            
    return adjusted_paragraphs_by_file



def adjust_paragraph_length(paragraphs, min_tokens, max_tokens):
    adjusted_paragraphs = paragraphs.copy()
    adjustments_needed = True
    max_iterations = 3
    current_iteration = 0

    while adjustments_needed and current_iteration < max_iterations:
        new_adjusted_paragraphs = []

        i = 0
        while i < len(adjusted_paragraphs):
            paragraph = adjusted_paragraphs[i]
            tokens = tokenizer(paragraph, add_special_tokens=False)['input_ids']

            if len(tokens) > max_tokens:
                split_idx = max_tokens
                while split_idx > 0 and tokens[split_idx] != tokenizer.sep_token_id:
                    split_idx -= 1
                new_adjusted_paragraphs.append(tokenizer.decode(tokens[:split_idx], skip_special_tokens=True))
                new_adjusted_paragraphs.append(tokenizer.decode(tokens[split_idx:], skip_special_tokens=True))
                i += 1
            elif len(tokens) < min_tokens and i < len(adjusted_paragraphs) - 1:
                next_tokens = tokenizer(adjusted_paragraphs[i + 1], add_special_tokens=False)['input_ids']
                combined_tokens = tokens + next_tokens

                if len(combined_tokens) <= max_tokens:
                    new_adjusted_paragraphs.append(tokenizer.decode(combined_tokens, skip_special_tokens=True))
                    i += 2  # Skip the next paragraph since it has been merged
                else:
                    new_adjusted_paragraphs.append(paragraph)
                    i += 1
            else:
                new_adjusted_paragraphs.append(paragraph)
                i += 1

        adjustments_needed = any(
            not min_tokens <= len(tokenizer(p, add_special_tokens=False)['input_ids']) <= max_tokens
            for p in new_adjusted_paragraphs
        )

        adjusted_paragraphs = new_adjusted_paragraphs
        current_iteration += 1

    # If there are still paragraphs longer than max_tokens, truncate them
    for i, paragraph in enumerate(adjusted_paragraphs):
        tokens = tokenizer(paragraph, add_special_tokens=False)['input_ids']
        if len(tokens) > max_tokens:
            adjusted_paragraphs[i] = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    return adjusted_paragraphs




# Set the input and output directories
input_folder = 'data/subset_paragraphs_filtered'
output_folder = 'data/subset_paragraphs_filtered_adjusted'

# Get a list of all txt files in the input folder
input_files = glob.glob(os.path.join(input_folder, '*.txt'))

# Read paragraphs from all input files
min_tokens = 300
max_tokens = 500
# Read and adjust paragraphs from all input files
adjusted_paragraphs_by_file = read_and_adjust_paragraphs_from_files(input_files, min_tokens, max_tokens)

# Save the adjusted paragraphs in the output folder
for input_file_path, adjusted_paragraphs in adjusted_paragraphs_by_file.items():
    basename = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_folder, basename)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(adjusted_paragraphs))

print(f"Adjusted paragraphs saved in {output_folder}")