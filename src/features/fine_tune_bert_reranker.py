from danlp.models import load_bert_base_model
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
#from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW
import os


import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error

class RerankDataset(Dataset):
    def __init__(self, tokenizer, questions, paragraphs, scores):
        self.tokenizer = tokenizer
        self.questions = questions
        self.paragraphs = paragraphs
        self.scores = scores

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        paragraph = self.paragraphs[idx]
        score = self.scores[idx]

        inputs = self.tokenizer.encode_plus(
            question,
            paragraph,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'score': torch.tensor(score, dtype=torch.float)
        }
    

def train_reranker(model, tokenizer, train_data, val_data, epochs=3, batch_size=16, learning_rate=2e-5, warmup_steps=0, weight_decay=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = RerankDataset(tokenizer, *train_data)
    val_dataset = RerankDataset(tokenizer, *val_data)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            model.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=scores)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                scores = batch['score'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=scores)
                loss = outputs.loss
                total_val_loss += loss.item()

                y_true.extend(scores.cpu().numpy())
                y_pred.extend(outputs.logits.squeeze().cpu().numpy())

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}, Validation RMSE: {val_rmse}")

    return model


# Instantiate the reranker
reranker = BertReranker()

# Train the reranker
train_data = (train_questions, train_paragraphs, train_scores)
val_data = (val_questions, val_paragraphs, val_scores)
trained_reranker = train_reranker(reranker.model, reranker.tokenizer, train_data, val_data)
