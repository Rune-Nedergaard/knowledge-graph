import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
from danlp.models import load_bert_base_model

class ParagraphDataset(Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.files = [f for f in os.listdir(file_dir) if f.endswith('.txt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.file_dir, self.files[idx]), 'r', encoding='utf-8') as f:
            paragraphs = f.read().split('\n')
        return self.files[idx], [p for p in paragraphs if p.strip()]


def collate_fn(batch):
    filenames, paragraphs_list = zip(*batch)
    return filenames, paragraphs_list


def embed_and_save(model, data_loader, device):
    embeddings = {}
    for filenames, paragraphs_list in tqdm(data_loader, desc='Embedding paragraphs'):
        for filename, paragraphs in zip(filenames, paragraphs_list):
            file_embeddings = []
            for index, paragraph in enumerate(paragraphs):
                _, embedding, = model.embed_text(paragraph)
                file_embeddings.append((index, embedding))
            embeddings[filename] = file_embeddings

    return embeddings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = load_bert_base_model()
model.model.to(device)
tokenizer = model.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



file_dir = 'data/all_paragraphs/paragraphs'
dataset = ParagraphDataset(file_dir)
data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

embeddings = embed_and_save(model, data_loader, device)

saved_with_h5py = False
try:
    with h5py.File('all_embeddings.h5', 'w') as f:
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
        with open('embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved to embeddings.pickle")
    except Exception as e:
        print("Failed to save embeddings with pickle. Error:", e)
