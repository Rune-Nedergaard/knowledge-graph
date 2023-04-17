import os
import glob
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW
from danlp.models import load_bert_base_model
from danlp.models.bert_models import BertBase as DaNLPBertBase
import numpy as np
import random
from transformers import PreTrainedTokenizerFast
from typing import List, Dict, Iterable
from torch import Tensor
from transformers import PreTrainedTokenizerFast
import nltk
import itertools

import spacy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.two_towers_fine_tune_multiplenegatives import *


def find_subfolders_with_prefix(folder_path, prefix):
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    return [os.path.join(folder_path, d) for d in subfolders if d.startswith(prefix)]


# Define the root directory to be scanned
root_dir = 'models'

# Find all relevant subfolders
#checkpoints_dirs = find_subfolders_with_prefix(root_dir, "two_tower_checkpoints_multiplenegatives")

checkpoints_dirs = ['models/two_tower_checkpoints_multiplenegatives_v4']

def latest_checkpoint(checkpoints_dir):
    # Find all checkpoint files
    files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))

    # Debug: print the file paths
    print("Files:", files)

    # Sort files by modification time
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    # Return the latest checkpoint file
    return files[0] if files else None




danish_bert_question = BertCustomBase(device)
danish_bert_paragraph = BertCustomBase(device)

tokenizer = danish_bert_question.tokenizer




class QuestionParagraphDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, positive_paragraph = self.data[idx]

        # Randomly select an unrelated paragraph as a negative sample
        negative_idx = random.choice([i for i in range(len(self.data)) if i != idx])
        _, negative_paragraph = self.data[negative_idx]

        return {
            "question": question,
            "positive_paragraph": positive_paragraph,
            "negative_paragraph": negative_paragraph
        }
    

def collate_fn(batch):
    question_batch = []
    positive_paragraph_batch = []
    negative_paragraph_batch = []

    for instance in batch:
        question_batch.append(instance["question"])
        
        # Tokenize paragraphs into sentences
        positive_paragraph_sentences = spacy_sent_tokenize(instance["positive_paragraph"])
        negative_paragraph_sentences = spacy_sent_tokenize(instance["negative_paragraph"])
        
        positive_paragraph_batch.append(positive_paragraph_sentences)
        negative_paragraph_batch.append(negative_paragraph_sentences)

    return {
        "question": question_batch,
        "positive_paragraph": positive_paragraph_batch,
        "negative_paragraph": negative_paragraph_batch
    }


# Load the dataset and create a dataloader
dataset = QuestionParagraphDataset("data/big_question_chunk_pairs.pkl")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
two_tower_model = TwoTowerSimilarityModel(danish_bert_question, danish_bert_paragraph)
custom_multiple_negatives_ranking_loss = CustomMultipleNegativesRankingLoss(two_tower_model)



# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Iterate over each directory in checkpoints_dirs
for checkpoints_dir in checkpoints_dirs:
    # Load the latest checkpoint
    latest_checkpoint_file = latest_checkpoint(checkpoints_dir)
    
    if latest_checkpoint_file:
        print(f"Loading latest checkpoint: {latest_checkpoint_file}")
        two_tower_model = torch.load(latest_checkpoint_file)

        # Validation loop
        two_tower_model.eval()
        epoch_val_loss = 0
        val_progress_bar = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for batch in val_progress_bar:
                loss = custom_multiple_negatives_ranking_loss(batch, None)
                epoch_val_loss += loss.item()
                val_progress_bar.set_description(f"Validation (loss: {loss.item():.4f})")

        epoch_val_loss /= len(val_dataloader)
        print(f"Validation loss: {epoch_val_loss:.4f}")
    else:
        print(f"No checkpoint found in {checkpoints_dir}.")
    
    try:
        #save the validation loss to a txt file that matches the checkpoint file
        with open(latest_checkpoint_file.replace('.pt', '.txt'), 'w') as f:
            f.write(f"Validation loss: {epoch_val_loss:.4f}")
    except:
        pass
