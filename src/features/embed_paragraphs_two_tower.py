import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.two_towers_fine_tune_seed_optimization import BertCustomBase, TwoTowerSimilarityModel, TwoTowerModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Load BaseBert for tokenization in dataset class get item
danish_bert_question = BertCustomBase(device)
tokenizer = danish_bert_question.tokenizer


class ParagraphDataset(Dataset):
    def __init__(self, file_dir, tokenizer=tokenizer, max_seq_len=512, device=device):
        self.file_dir = file_dir
        self.files = [f for f in os.listdir(file_dir) if f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.file_dir, self.files[idx]), 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
        
        tokenized_paragraphs = []
        attention_masks = []
        for p in paragraphs:
            if p.strip():
                inputs = self.tokenizer.encode_plus(p, return_tensors='pt', max_length=self.max_seq_len, truncation=True, padding='max_length')
                input_ids = inputs['input_ids'].squeeze(0).to(self.device)
                attention_mask = inputs['attention_mask'].squeeze(0).to(self.device)
                tokenized_paragraphs.append(input_ids)
                attention_masks.append(attention_mask)
        
        return self.files[idx], tokenized_paragraphs, attention_masks


def collate_fn(batch):
    filenames, paragraphs_list, attention_masks_list = zip(*batch)
    return filenames, paragraphs_list, attention_masks_list


def embed_and_save(model, data_loader, device):
    embeddings = {}
    for filenames, paragraphs_list, attention_masks_list in tqdm(data_loader, desc='Embedding paragraphs'):
        for filename, paragraphs, attention_masks in zip(filenames, paragraphs_list, attention_masks_list):
            file_embeddings = []
            for index, (paragraph, attention_mask) in enumerate(zip(paragraphs, attention_masks)):
                inputs = {'input_ids': paragraph.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
                with torch.no_grad():
                    paragraph_embedding = model.two_tower_model.forward_paragraph(**inputs)
                file_embeddings.append((index, paragraph_embedding))
            embeddings[filename] = file_embeddings

    return embeddings



# Load the TwoTowerModel
two_tower_model_path = 'models/fine_tuned_model_two_tower/model.pt'

# Initialize TwoTowerSimilarityModel
model = torch.load(two_tower_model_path)

file_dir = 'data/subset_paragraphs_filtered'
dataset = ParagraphDataset(file_dir)
data_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

embeddings = embed_and_save(model, data_loader, device)

saved_with_h5py = False
try:
    with h5py.File('two_tower_embeddings.h5', 'w') as f:
        for filename, file_embeddings in embeddings.items():
            group = f.create_group(filename)
            for index, embedding in file_embeddings:
                group.create_dataset(str(index), data=embedding.cpu().numpy())
    print('Embeddings saved to embeddings.h5')
    saved_with_h5py = True
except Exception as e:
    print("Failed to save embeddings with h5py. Error:", e)

if not saved_with_h5py:
    try:
        with open('two_tower.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved to embeddings.pickle")
    except Exception as e:
        print("Failed to save embeddings with pickle. Error:", e)