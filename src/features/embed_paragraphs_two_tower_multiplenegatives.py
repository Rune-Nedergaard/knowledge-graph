import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.two_towers_fine_tune_multiplenegatives import *
import torch.multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
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
                    # Tokenize the paragraph into sentences
                    sentences = spacy_sent_tokenize(paragraph)

                    # Embed each sentence
                    sentence_embeddings = []
                    for sentence in sentences:
                        sentence_embedding = model.paragraph_model.embed_text(sentence)
                        sentence_embeddings.append(sentence_embedding)

                    # Stack and mean-pool the sentence embeddings
                    stacked_embeddings = torch.stack(sentence_embeddings, dim=0)
                    paragraph_embedding = torch.mean(stacked_embeddings, dim=0)

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

two_tower_model_path = 'models/two_tower_checkpoints_multiplenegatives_v4/model_step_84390_epoch_1.pt'

model = torch.load(two_tower_model_path)

file_dir = 'data/subset_paragraphs_filtered' ##### CHANGE THIS TO THE CORRECT FOLDER DOING THE FINAL RUN ########
dataset = ParagraphDataset(file_dir)
data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=0, shuffle=True)

embeddings_folder = 'data/embeddings'
embed_and_save(model, data_loader, device, embeddings_folder)
embeddings = load_embeddings_from_folder(embeddings_folder)

try:
    with h5py.File('two_tower_embeddings.h5', 'w') as f:
        for filename, file_embeddings in embeddings.items():
            group = f.create_group(filename)
            for index, embedding in file_embeddings:
                group.create_dataset(str(index), data=embedding.cpu().numpy())
    print('Embeddings saved to embeddings.h5')
except Exception as e:
    print("Failed to save embeddings with h5py. Error:", e)

    try:
        with open('two_tower.pickle', 'wb') as f:
            pickle.dump(embeddings, f)
        print("Embeddings saved to embeddings.pickle")
    except Exception as e:
        print("Failed to save embeddings with pickle. Error:", e)