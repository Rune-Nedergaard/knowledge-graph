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

import spacy
from spacy.cli import download as spacy_download
spacy_download('da_core_news_sm')


nlp = spacy.load("da_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
nlp.add_pipe("sentencizer")

def spacy_sent_tokenize(text):
    return [sent.text for sent in nlp(text).sents]


# Create a directory to save checkpoints
checkpoints_dir = 'models/two_tower_checkpoints_multiplenegatives_v7'
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
        # Split the paragraph into sentences
        sentences = spacy_sent_tokenize(text)


        # Embed each sentence and get the [CLS] token representation
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embedding = self.embed_text(sentence)
            sentence_embeddings.append(sentence_embedding)

        # Stack and mean-pool the [CLS] token representations of the sentences
        stacked_embeddings = torch.stack(sentence_embeddings, dim=0)
        pooled_embedding = torch.mean(stacked_embeddings, dim=0)

        return pooled_embedding      
    
    #exposing the model parameters to the optimizer
    @property
    def model_parameters(self):
        return self.model.parameters()
        

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
    def __init__(self, question_model: BertCustomBase, paragraph_model: BertCustomBase):
        super(TwoTowerSimilarityModel, self).__init__()
        self.question_model = question_model
        self.paragraph_model = paragraph_model

    def forward(self, questions: List[str], paragraphs: List[List[str]]):
        # Embed the questions and paragraphs each element in batch.
        # Notice that the paragraphs contain multiple sentences each that are embedded separately and then pooled
        question_embeddings = torch.stack([self.question_model.embed_text(question) for question in questions])
        
        paragraph_sentence_embeddings = [
            [self.paragraph_model.embed_text(sentence) for sentence in paragraph_sentences]
            for paragraph_sentences in paragraphs
        ]

        # Average the embeddings of all sentences in a paragraph to get a single paragraph embedding for each paragraph
        paragraph_embeddings = [
            torch.stack(sentence_embeddings, dim=0).mean(dim=0)
            for sentence_embeddings in paragraph_sentence_embeddings
        ]

        # Return the question embeddings and paragraph embeddings as tensors
        return question_embeddings, torch.stack(paragraph_embeddings)
    
    def parameters(self):
        return itertools.chain(self.question_model.model_parameters, self.paragraph_model.model_parameters)



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



two_tower_model = TwoTowerSimilarityModel(danish_bert_question, danish_bert_paragraph)
custom_multiple_negatives_ranking_loss = CustomMultipleNegativesRankingLoss(two_tower_model)

# Define optimizer and learning rate scheduler
optimizer = AdamW(two_tower_model.parameters(), lr=1e-6, weight_decay=0.005)
print(f"Using optimizer with the following parameters:")
print(optimizer.defaults)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)


# Define number of epochs
num_epochs = 10

best_val_loss = float("inf")
train_losses = []
val_losses = []

from transformers import get_linear_schedule_with_warmup

# Define the number of training steps and warmup steps for the scheduler
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(num_training_steps * 0.1)  # 10% of total training steps for the first epoch

# Create the scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Training loop
num_steps_to_save = len(train_dataloader) // 5  # Save every 20% of the training data
step_counter = 0
running_train_loss = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    two_tower_model.train()
    epoch_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc="Training")
    for batch_idx, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()
        loss = custom_multiple_negatives_ranking_loss(batch, None)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        running_train_loss += loss.item()
        train_progress_bar.set_description(f"Training (loss: {loss.item():.4f})")

        # Update the scheduler on the first epoch after warmup
        if epoch == 0 and scheduler.get_last_lr()[0] != optimizer.defaults['lr']:
            scheduler.step()

        # Increment the step counter and save the model and loss if necessary
        step_counter += 1
        if step_counter == num_steps_to_save:
            step_counter = 0
            save_path = os.path.join(checkpoints_dir, f"model_step_{batch_idx+1}_epoch_{epoch+1}.pt")
            torch.save(two_tower_model, save_path)
            print(f"Model saved at step {batch_idx+1} of epoch {epoch+1}")

            # Save average loss since the last checkpoint
            average_loss_since_last_checkpoint = running_train_loss / num_steps_to_save
            with open(os.path.join(checkpoints_dir, f"average_loss_step_{batch_idx+1}_epoch_{epoch+1}.pkl"), "wb") as f:
                pickle.dump(average_loss_since_last_checkpoint, f)
            running_train_loss = 0

    epoch_train_loss /= len(train_dataloader)
    print(f"Train loss: {epoch_train_loss:.4f}")
    train_losses.append(epoch_train_loss)

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
    val_losses.append(epoch_val_loss)

    # Update the learning rate scheduler based on validation loss
    scheduler.step(epoch_val_loss)

    # Save the model (not just the state_dict) whenever the validation loss is better than the last epoch
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_path = os.path.join(checkpoints_dir, f"best_model_epoch_{epoch+1}.pt")
        torch.save(two_tower_model, best_model_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    # Save train and validation losses as pickles every epoch
    with open(os.path.join(checkpoints_dir, f"train_losses_epoch_{epoch+1}.pkl"), "wb") as f:
        pickle.dump(train_losses, f)

    with open(os.path.join(checkpoints_dir, f"val_losses_epoch_{epoch+1}.pkl"), "wb") as f:
        pickle.dump(val_losses, f)
