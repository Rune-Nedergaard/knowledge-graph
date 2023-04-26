import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from src.features.two_towers_fine_tune_multiplenegatives import *
from two_towers_for_embedding_import_v14 import create_two_tower_model_and_loss, TwoTowerSimilarityModel, BertCustomBase
import torch.multiprocessing as mp
import dill



danish_bert, two_tower_model, loss_function = create_two_tower_model_and_loss()

globals()['TwoTowerSimilarityModel'] = TwoTowerSimilarityModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

tokenizer = danish_bert.tokenizer


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
        for p in paragraphs:
            if p.strip():
                tokenized_paragraphs.append(p)
        
        return self.files[idx], tokenized_paragraphs
        


def collate_fn(batch):
    filenames, paragraphs_list = zip(*batch)
    return filenames, paragraphs_list

def embedding_file_exists(folder, filename, index):
    filepath = os.path.join(folder, f"{filename}_{index}.npy")
    return os.path.isfile(filepath)

def save_embedding_to_folder(folder, filename, index, embedding):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the embedding as a .npy file
    filepath = os.path.join(folder, f"{filename}_{index}.npy")
    torch.save(embedding.detach().cpu().numpy(), filepath)  # Detach the tensor before converting to a numpy array


def embed_and_save(model, data_loader, device, embeddings_folder):
    for filenames, paragraphs_list in tqdm(data_loader, desc='Embedding paragraphs'):
        for filename, paragraphs in zip(filenames, paragraphs_list):
            for index, paragraph in enumerate(paragraphs):
                if not embedding_file_exists(embeddings_folder, filename, index):
                    # Embed the paragraph
                    paragraph_embedding = model.embed_paragraph(paragraph)  # Using the shared model weights

                    save_embedding_to_folder(embeddings_folder, filename, index, paragraph_embedding)


def load_embeddings_from_folder(embeddings_folder):
    embeddings = {}
    for file in os.listdir(embeddings_folder):
        if file.endswith(".npy"):
            filename, index = file[:-4].rsplit('_', 1)
            index = int(index)
            if filename not in embeddings:
                embeddings[filename] = []
            embeddings[filename].append((index, torch.from_numpy(torch.load(os.path.join(embeddings_folder, file)))))

    for filename in embeddings.keys():
        embeddings[filename].sort(key=lambda x: x[0])

    return embeddings

two_tower_model_path = 'models/two_tower_checkpoints_multiplenegatives_v14/best_model_epoch_3.pt'


best_model_path = 'models/two_tower_checkpoints_multiplenegatives_v14/best_model_epoch_3.pt'
print("Best model path:", best_model_path)
print("Path exists:", os.path.exists(best_model_path))
print("Directory contents:", os.listdir(os.path.dirname(best_model_path)))



model = torch.load(best_model_path, pickle_module=dill)
model.to(device)
print(model)
print(model.state_dict())


file_dir = 'data/all_paragraphs/paragraphs' ##### CHANGE THIS TO THE CORRECT FOLDER DOING THE FINAL RUN ########
dataset = ParagraphDataset(file_dir)
data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=True)

embeddings_folder = 'data/embeddings_v14'
embed_and_save(model, data_loader, device, embeddings_folder)
embeddings = load_embeddings_from_folder(embeddings_folder)

try:
    with h5py.File('v14_two_tower_embeddings.h5', 'w') as f:
        for filename, file_embeddings in embeddings.items():
            group = f.create_group(filename)
            for index, embedding in file_embeddings:
                group.create_dataset(str(index), data=embedding.cpu().numpy())
    print('Embeddings saved to embeddings.h5')
except Exception as e:
    print("Failed to save embeddings with h5py. Error:", e)

    try:
        with open('final_two_tower.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved to embeddings.pickle")
    except Exception as e:
        print("Failed to save embeddings with pickle. Error:", e)