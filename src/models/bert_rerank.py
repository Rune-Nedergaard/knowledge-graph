import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error

class BertReranker(BertBase):
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        super().__init__(cache_dir, verbose)
        self.model = BertForSequenceClassification.from_pretrained(
            self.path_model,
            num_labels=1
        )
        self.model.eval()

    def predict(self, questions, paragraphs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        dataset = RerankDataset(self.tokenizer, questions, paragraphs, [0] * len(questions))
        dataloader = DataLoader(dataset, batch_size=16)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze().cpu().numpy()
                predictions.extend(logits)

        return predictions
