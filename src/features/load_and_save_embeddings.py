import os
import h5py
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from src.features.two_towers_fine_tune_multiplenegatives import *
import torch.multiprocessing as mp
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#danish_bert_question = BertCustomBase(device)
#tokenizer = danish_bert_question.tokenizer


def load_embeddings_from_folder(embeddings_folder):
    embeddings = {}
    for file in tqdm(os.listdir(embeddings_folder), desc="Loading embeddings"):
        if file.endswith(".npy"):
            filepath = os.path.join(embeddings_folder, file)
            if os.path.getsize(filepath) > 0:  # Check if the file is not empty
                try:
                    filename, index = file[:-4].rsplit('_', 1)
                    index = int(index)
                    if filename not in embeddings:
                        embeddings[filename] = []
                    embeddings[filename].append((index, torch.from_numpy(torch.load(filepath))))
                except Exception as e:
                    print(f"Error loading file {file}: {e}")
            else:
                print(f"Skipping empty file {file}")

    for filename in embeddings.keys():
        embeddings[filename].sort(key=lambda x: x[0])

    return embeddings


#two_tower_model_path = 'models/two_tower_checkpoints_multiplenegatives_v4/model_step_84390_epoch_1.pt'

#model = torch.load(two_tower_model_path)

embeddings_folder = 'data/embeddings_final'

embeddings = load_embeddings_from_folder(embeddings_folder)

try:
    with h5py.File('final_two_tower_embeddings.h5', 'w') as f:
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