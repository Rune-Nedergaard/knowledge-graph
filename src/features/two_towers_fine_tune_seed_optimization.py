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
from danlp.models.bert_models import BertBase as DaNLPBertBase
import numpy as np
import random
# Create a directory to save checkpoints
checkpoints_dir = 'models/two_tower_checkpoints'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Set device and check for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
    print(f"Using device: {device}")

"""
Inserting BertBase with modifications here, in order not to interfere with the original fine tuning script
"""





class BertCustomBase(DaNLPBertBase):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model.to(device)

    def embed_text(self, text):
  
        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize sentence with the BERT tokenizer
        tokenized_text = self.tokenizer.tokenize(marked_text)[:511]


        # Map the token strings to their vocabulary indeces
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)


        # Mark each of the tokens as belonging to sentence "1"
        segments_ids = [1] * len(tokenized_text)

         # Convert inputs to PyTorch tensors ### CHANGE RIGHT HERE ####
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1. to tokens x layers x embedding
        token_embeddings = token_embeddings.permute(1,0,2)

        # choose to concatenate last four layers, dim 4x 768 = 3072
        token_vecs_cat= [torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) for token in token_embeddings]
        # drop the CLS and the SEP tokens and embedding
        token_vecs_cat=token_vecs_cat[1:-1]
        tokenized_text =tokenized_text[1:-1]

        # chose to summarize the last four layers
        #token_vecs_sum=[torch.sum(token[-4:], dim=0) for token in token_embeddings]

        # sentence embedding
        # Calculate the average of all token vectors for the second last layers
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

        return token_vecs_cat, sentence_embedding, tokenized_text
    

danish_bert_question = BertCustomBase(device)
danish_bert_paragraph = BertCustomBase(device)

tokenizer = danish_bert_question.tokenizer
bert_model_question = danish_bert_question.model.to(device)
bert_model_paragraph = danish_bert_paragraph.model.to(device)

dataset = pickle.load(open('data/fine_tune_dataset.pkl', 'rb'))
labels = pickle.load(open('data/fine_tune_labels.pkl', 'rb'))

class TwoTowerModel(nn.Module):
    def __init__(self, question_model, paragraph_model):
        super().__init__()
        self.question_model = question_model
        self.paragraph_model = paragraph_model
        
    def forward_question(self, input_ids, attention_mask):
        outputs = self.question_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        return pooled_output

    def forward_paragraph(self, input_ids, attention_mask):
        outputs = self.paragraph_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        return pooled_output
    
class QuestionChunkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len=512, device=device):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        question, paragraph = self.texts[idx]

        question_tokenized = self.tokenizer.tokenize(question)
        paragraph_tokenized = self.tokenizer.tokenize(paragraph)

        question_input_ids = self.tokenizer.convert_tokens_to_ids(question_tokenized)
        paragraph_input_ids = self.tokenizer.convert_tokens_to_ids(paragraph_tokenized)

        max_seq_len = self.max_seq_len
        question_input_ids = question_input_ids[:max_seq_len]
        paragraph_input_ids = paragraph_input_ids[:max_seq_len]


        # Calculate the maximum sequence length for question and paragraph separately
        max_question_seq_len = min(len(question_input_ids), self.max_seq_len)
        max_paragraph_seq_len = min(len(paragraph_input_ids), self.max_seq_len)

        # Truncate or pad the token ids for question and paragraph separately
        question_input_ids = question_input_ids[:max_question_seq_len] + [self.tokenizer.pad_token_id] * (self.max_seq_len - max_question_seq_len)
        paragraph_input_ids = paragraph_input_ids[:max_paragraph_seq_len] + [self.tokenizer.pad_token_id] * (self.max_seq_len - max_paragraph_seq_len)

        # Create attention masks
        question_attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in question_input_ids]
        paragraph_attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in paragraph_input_ids]

        # Move tensors to the correct device
        question_input_ids = torch.tensor(question_input_ids).to(self.device)
        question_attention_mask = torch.tensor(question_attention_mask).to(self.device)
        paragraph_input_ids = torch.tensor(paragraph_input_ids).to(self.device)
        paragraph_attention_mask = torch.tensor(paragraph_attention_mask).to(self.device)

        return {
            'question_input_ids': question_input_ids,
            'question_attention_mask': question_attention_mask,
            'paragraph_input_ids': paragraph_input_ids,
            'paragraph_attention_mask': paragraph_attention_mask,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device),
        }


    
    
class TwoTowerSimilarityModel(nn.Module):
    def __init__(self, two_tower_model):
        super().__init__()
        self.two_tower_model = two_tower_model
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask, return_embeddings=False):
        question_embedding = self.two_tower_model.forward_question(question_input_ids, question_attention_mask)
        paragraph_embedding = self.two_tower_model.forward_paragraph(paragraph_input_ids, paragraph_attention_mask)

        if return_embeddings:
            return question_embedding, paragraph_embedding

        cosine_sim = self.cosine_similarity(question_embedding, paragraph_embedding)
        return cosine_sim


if __name__ == "__main__":

     # Define dataset
    question_chunk_dataset = QuestionChunkDataset(dataset, labels, tokenizer=danish_bert_question.tokenizer, max_seq_len=512, device=device)
    
    #Define loss
    loss_function = nn.BCELoss()
    # Split your data into training, validation, and test sets
    train_size = int(0.8 * len(question_chunk_dataset))
    val_size = int(0.1 * len(question_chunk_dataset))
    test_size = len(question_chunk_dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(question_chunk_dataset, [train_size, val_size, test_size])

    # Create DataLoaders for the three sets
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False,  drop_last=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False,  drop_last=True)


    num_seeds = 15
    steps_per_seed = 3000
    steps_for_testing = 1000

    best_seed = None
    best_loss = float('inf')

    # Finding the best seed
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        two_tower_model = TwoTowerModel(bert_model_question, bert_model_paragraph).to(device)
        two_tower_similarity_model = TwoTowerSimilarityModel(two_tower_model).to(device)

        # Define the loss function and optimizer
        loss_function = nn.BCELoss()
        optimizer = AdamW(two_tower_similarity_model.parameters(), lr=2e-5, weight_decay=0.01)

        print(f"Testing seed {seed + 1}/{num_seeds}")
        two_tower_similarity_model.train()
        train_loss = 0
        steps = 0

        progress_bar = tqdm(train_loader, desc="Training", total=steps_per_seed)

        for batch in progress_bar:
            optimizer.zero_grad()
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            paragraph_input_ids = batch['paragraph_input_ids'].to(device)
            paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            similarities = two_tower_similarity_model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
            loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(two_tower_similarity_model.parameters(), max_norm=1.0)  # Added Gradient clipping
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_description(f"Training (loss: {loss.item():.4f})")

            steps += 1
            if steps >= steps_per_seed:
                break

        train_loss /= steps

        # Test loop
        two_tower_similarity_model.eval()
        test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Testing", total=steps_for_testing)
            for batch in progress_bar:
                question_input_ids = batch['question_input_ids'].to(device)
                question_attention_mask = batch['question_attention_mask'].to(device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                labels = batch['label'].to(device)

                similarities = two_tower_similarity_model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss

                test_loss += loss.item()
                progress_bar.set_description(f"Testing (loss: {loss.item():.4f})")

                test_steps += 1
                if test_steps >= steps_for_testing:
                    break

        test_loss /= len(val_loader)

        print(f"Seed {seed + 1} Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_seed = seed

    print(f"Best seed:{best_seed} with Test Loss: {best_loss:.4f}")


    two_tower_model = TwoTowerModel(bert_model_question, bert_model_paragraph).to(device)
    two_tower_similarity_model = TwoTowerSimilarityModel(two_tower_model).to(device)

    # Define the optimizer
    
    optimizer = AdamW(two_tower_similarity_model.parameters(), lr=2e-5, weight_decay=0.01)

 



    start_epoch = 1  # Set the epoch number from which you want to continue training (1-indexed)


    train_losses, test_losses = [], []
    #Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

    # Fine-tune the model
    num_epochs = 10
    print(f"Fine-tuning the model for {num_epochs} epochs...")

    best_test_loss = float('inf')  # Initialize the best validation loss as infinity


    for epoch in range(start_epoch - 1, num_epochs):  # Subtract 1 to make it zero-indexed

        print(f"Epoch {epoch + 1}/{num_epochs}")
        two_tower_similarity_model.train()
        train_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            paragraph_input_ids = batch['paragraph_input_ids'].to(device)
            paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            similarities = two_tower_similarity_model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
            loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(two_tower_similarity_model.parameters(), max_norm=1.0)  # Added Gradient clipping
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_description(f"Training (loss: {loss.item():.4f})")



        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Test loop
        two_tower_similarity_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Testing")
            for batch in progress_bar:
                question_input_ids = batch['question_input_ids'].to(device)
                question_attention_mask = batch['question_attention_mask'].to(device)
                paragraph_input_ids = batch['paragraph_input_ids'].to(device)
                paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
                labels = batch['label'].to(device)

                similarities = two_tower_similarity_model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
                loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss

                test_loss += loss.item()
                progress_bar.set_description(f"Testing (loss: {loss.item():.4f})")

        test_loss /= len(val_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}")
        
        # Call the scheduler.step() function with the test_loss
        scheduler.step(test_loss)

        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': two_tower_similarity_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_losses,
            'test_loss_history': test_losses,
        }, checkpoint_path)

        #save test loss and train loss
        import pickle
        with open('data/train_losses_two_tower.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open('data/test_losses_two_tower.pkl', 'wb') as f:
            pickle.dump(test_losses, f)


        # Save the model if the test loss is lower than the best test loss
        if test_loss < best_test_loss:
            print(f"Improved test loss from {best_test_loss:.4f} to {test_loss:.4f}. Saving the model.")
            best_test_loss = test_loss
            save_directory = 'models/fine_tuned_model_two_tower'
            os.makedirs(save_directory, exist_ok=True)
            torch.save(two_tower_similarity_model, os.path.join(save_directory, 'model.pt'))
            try:
                tokenizer.save_pretrained(save_directory)
            except:
                pass

        
