import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the distiluse-base-multilingual-cased-v2 model
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2').to(device)


class ParagraphDataset(Dataset):
    def __init__(self, file_dir, max_seq_len=512, device=device):
        self.file_dir = file_dir
        self.files = [f for f in os.listdir(file_dir) if f.endswith('.txt')]
        self.max_seq_len = max_seq_len
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.file_dir, self.files[idx]), 'r', encoding='iso-8859-1') as f:
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
                    # Embed the paragraph using the distiluse-base-multilingual-cased-v2 model
                    paragraph_embedding = model.encode(paragraph, device=device)
                    save_embedding_to_folder(embeddings_folder, filename, index, torch.tensor(paragraph_embedding))

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

if __name__ == '__main__':
    file_dir = 'data/questions_rephrased' ##### CHANGE THIS TO THE CORRECT FOLDER DOING THE FINAL RUN ########
    dataset = ParagraphDataset(file_dir)
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=0, shuffle=True)

    embeddings_folder = 'data/embeddings_sbert'
    embed_and_save(sentence_model, data_loader, device, embeddings_folder)
    embeddings = load_embeddings_from_folder(embeddings_folder)

    try:
        with h5py.File('final_distiluse_embeddings.h5', 'w') as f:
            for filename, file_embeddings in embeddings.items():
                group = f.create_group(filename)
                for index, embedding in file_embeddings:
                    group.create_dataset(str(index), data=embedding.cpu().numpy())
        print('Embeddings saved to final_distiluse_embeddings.h5')
    except Exception as e:
        print("Failed to save embeddings with h5py. Error:", e)

        try:
            with open('final_distiluse_embeddings.pickle', 'wb') as f:
                pickle.dump(embeddings, f)
            print("Embeddings saved to final_distiluse_embeddings.pickle")
        except Exception as e:
            print("Failed to save embeddings with pickle. Error:", e)