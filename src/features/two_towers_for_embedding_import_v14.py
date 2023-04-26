import os
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
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

#import spacy
#from spacy.cli import download as spacy_download
#spacy_download('da_core_news_sm')


#nlp = spacy.load("da_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
#nlp.add_pipe("sentencizer")

#def spacy_sent_tokenize(text):
#    return [sent.text for sent in nlp(text).sents]


# Create a directory to save checkpoints
checkpoints_dir = 'models/two_tower_checkpoints_multiplenegatives_v14'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Set device and check for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
    print(f"Using device: {device}")

"""
Inserting BertBase with modifications here, in order not to interfere with the original fine tuning script
"""


class BertCustomBase(DaNLPBertBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model.to(device)

    def embed_text(self, text):
            marked_text = "[CLS] " + text + " [SEP]"
            # Tokenize sentence with the BERT tokenizer
            tokenized_text = self.tokenizer.tokenize(marked_text)[:511]

            # Map the token strings to their vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            # Mark each of the tokens as belonging to sentence "1"
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            segments_tensors = torch.tensor([segments_ids]).to(self.device)

            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

            # Sentence embedding: Calculate the average of all token vectors for the second last layer
            sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

            return sentence_embedding
    
    def embed_question(self, text):
        # Reusing the existing embed_text method for questions
        sentence_embedding = self.embed_text(text)
        return sentence_embedding

    def embed_paragraph(self, text):
        # Embed the entire paragraph directly, without splitting it into sentences
        paragraph_embedding = self.embed_text(text)
        return paragraph_embedding
        
        
    #exposing the model parameters to the optimizer
    @property
    def model_parameters(self):
        return self.model.parameters()
        

danish_bert = BertCustomBase(device)
tokenizer = danish_bert.tokenizer





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
        positive_paragraph_batch.append(instance["positive_paragraph"])
        negative_paragraph_batch.append(instance["negative_paragraph"])

    return {
        "question": question_batch,
        "positive_paragraph": positive_paragraph_batch,
        "negative_paragraph": negative_paragraph_batch
    }



def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))



class TwoTowerSimilarityModel(nn.Module):
    def __init__(self, bert_model: BertCustomBase):
        super(TwoTowerSimilarityModel, self).__init__()
        self.bert_model = bert_model

    def forward(self, questions: List[str], paragraphs: List[str]):
        question_embeddings = torch.stack([self.bert_model.embed_text(question) for question in questions])
        paragraph_embeddings = torch.stack([self.bert_model.embed_text(paragraph) for paragraph in paragraphs])

        return question_embeddings, paragraph_embeddings
    
    def embed_paragraph(self, text):
        return self.bert_model.embed_paragraph(text)

    def parameters(self):
        return self.bert_model.model_parameters


class CustomMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: TwoTowerSimilarityModel, scale: float = 20.0, similarity_fct=cos_sim):
        super(CustomMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Tensor], labels: Tensor):
        questions = batch["question"]
        positive_paragraphs = batch["positive_paragraph"]
        negative_paragraphs = batch["negative_paragraph"]
        
        question_embeddings, positive_paragraph_embeddings = self.model(questions, positive_paragraphs)
        _, negative_paragraph_embeddings = self.model(questions, negative_paragraphs)
        
        positive_scores = self.similarity_fct(question_embeddings, positive_paragraph_embeddings) * self.scale
        negative_scores = self.similarity_fct(question_embeddings, negative_paragraph_embeddings) * self.scale

        # Concatenate positive and negative scores along the columns
        scores = torch.cat((positive_scores, negative_scores), dim=1)

        
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)
    

def create_two_tower_model_and_loss():
    bert_model = BertCustomBase(device)
    two_tower_model = TwoTowerSimilarityModel(bert_model)
    loss_function = CustomMultipleNegativesRankingLoss(two_tower_model)
    return bert_model, two_tower_model, loss_function


if __name__ == "__main__":
        # Load the dataset and create a dataloader
        print("activated main")