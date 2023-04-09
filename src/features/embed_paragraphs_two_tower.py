import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
from src.models.bert_embed import BertEmbed
from src.features.two_towers_fine_tune import BertCustomBase, TwoTowerSimilarityModel, TwoTowerModel

# Load the pre-trained and fine-tuned model
save_directory = 'models/fine_tuned_model_two_tower'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the Danish BERT question model (BertCustomBase)
bert_model_question = BertCustomBase(device)
bert_model_paragraph = BertCustomBase(device)

# Initialize the TwoTowerModel
two_tower_model = TwoTowerModel(bert_model_question, bert_model_paragraph).to(device)

# Initialize the TwoTowerSimilarityModel
two_tower_similarity_model = TwoTowerSimilarityModel(two_tower_model).to(device)

# Load the state dict
state_dict = torch.load(os.path.join(save_directory, 'model.pt'))
two_tower_similarity_model.load_state_dict(state_dict)

# Move the model to the device
two_tower_similarity_model.to(device)


# Define a function to encode text using the model
def encode_input(text, bert_model):
    _, _, tokenized_text = bert_model.embed_text(text)
    input_ids = bert_model.tokenizer.encode(tokenized_text, return_tensors='pt', is_split_into_words=True).to(bert_model.device)
    attention_mask = input_ids != bert_model.tokenizer.pad_token_id
    return input_ids, attention_mask

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

def embed_and_save(two_tower_similarity_model, data_loader, device):
    embeddings = {}
    for filenames, paragraphs_list in tqdm(data_loader, desc='Embedding paragraphs'):
        for filename, paragraphs in zip(filenames, paragraphs_list):
            file_embeddings = []
            for index, paragraph in enumerate(paragraphs):
                input_ids, attention_mask = encode_input(paragraph, two_tower_similarity_model.two_tower_model.paragraph_model)
                _, paragraph_embedding = two_tower_similarity_model(None, None, input_ids, attention_mask, return_embeddings=True)
                file_embeddings.append((index, paragraph_embedding.cpu().detach().numpy()))
            embeddings[filename] = file_embeddings

    # Save the embeddings
    with open('paragraph_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

input_folder = 'path/to/your/input/folder'
paragraph_dataset = ParagraphDataset(input_folder)
data_loader = DataLoader(paragraph_dataset, batch_size=1, collate_fn=collate_fn)

embeddings = embed_and_save(two_tower_similarity_model, data_loader, device)
