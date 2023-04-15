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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

danish_bert = load_bert_base_model()

tokenizer = danish_bert.tokenizer
bert_model = danish_bert.model

dataset = pickle.load(open('data/fine_tune_dataset.pkl', 'rb'))
labels = pickle.load(open('data/fine_tune_labels.pkl', 'rb'))



class SemanticSimilarityModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # He initialization
        init.kaiming_normal_(self.linear.weight)

        # Xavier initialization
        # init.xavier_normal_(self.linear.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        logits = self.linear(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities.squeeze()
    
# Use DataParallel if multiple GPUs are available
model = SemanticSimilarityModel(bert_model)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

def load_checkpoint(model, optimizer, checkpoints_dir, epoch):
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        #removing the module prefix is necessary when the model was fine tuned on multuple GPUs
        model_state_dict = remove_module_prefix(checkpoint['model_state_dict'])
        
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        train_losses_path = 'data/train_losses.pkl'
        test_losses_path = 'data/test_losses.pkl'

        if os.path.exists(train_losses_path) and os.path.exists(test_losses_path):
            with open(train_losses_path, 'rb') as f:
                train_losses = pickle.load(f)
            with open(test_losses_path, 'rb') as f:
                test_losses = pickle.load(f)

            return train_losses, test_losses
        else:
            print(f"Train and test losses not found at: {train_losses_path} and {test_losses_path}")
            return [], []

    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        return [], []


checkpoints_dir = 'models/checkpoint_highbatch'

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)

start_epoch = 2  # Set the epoch number from which you want to continue training (1-indexed)

if start_epoch > 1:
    train_losses, test_losses = load_checkpoint(model, optimizer, checkpoints_dir, start_epoch)
else:
    train_losses, test_losses = [], []


torch.save(model.state_dict(), 'models/fine_tuned_model/check_highbatch{start_epoch}_model.pth'.format(start_epoch=start_epoch))