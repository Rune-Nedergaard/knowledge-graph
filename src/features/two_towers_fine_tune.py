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
    

danish_bert = BertCustomBase(device)

tokenizer = danish_bert.tokenizer
bert_model = danish_bert.model.to(device)

dataset = pickle.load(open('data/fine_tune_dataset.pkl', 'rb'))
labels = pickle.load(open('data/fine_tune_labels.pkl', 'rb'))

class TwoTowerModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        transformed = self.linear(pooled_output)
        return transformed

class QuestionChunkDataset(Dataset):
    def __init__(self, texts, labels, bert_model, max_seq_len=512, device=device):
        self.texts = texts
        self.labels = labels
        self.bert_model = bert_model
        self.max_seq_len = max_seq_len
        self.device = device
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        question, paragraph = self.texts[idx]

        question_tokenized, _, _ = self.bert_model.embed_text(question)
        paragraph_tokenized, _, _ = self.bert_model.embed_text(paragraph)

        question_input_ids = self.bert_model.tokenizer.convert_tokens_to_ids(question_tokenized)
        paragraph_input_ids = self.bert_model.tokenizer.convert_tokens_to_ids(paragraph_tokenized)

        max_seq_len = self.max_seq_len
        question_input_ids = question_input_ids[:max_seq_len]
        paragraph_input_ids = paragraph_input_ids[:max_seq_len]


        # Calculate the maximum sequence length for question and paragraph separately
        max_question_seq_len = min(len(question_input_ids), self.max_seq_len)
        max_paragraph_seq_len = min(len(paragraph_input_ids), self.max_seq_len)

        # Truncate or pad the token ids for question and paragraph separately
        question_input_ids = question_input_ids[:max_question_seq_len] + [self.bert_model.tokenizer.pad_token_id] * (self.max_seq_len - max_question_seq_len)
        paragraph_input_ids = paragraph_input_ids[:max_paragraph_seq_len] + [self.bert_model.tokenizer.pad_token_id] * (self.max_seq_len - max_paragraph_seq_len)

        # Create attention masks
        question_attention_mask = [1 if token_id != self.bert_model.tokenizer.pad_token_id else 0 for token_id in question_input_ids]
        paragraph_attention_mask = [1 if token_id != self.bert_model.tokenizer.pad_token_id else 0 for token_id in paragraph_input_ids]

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

    def forward(self, question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask):
        question_embedding = self.two_tower_model(question_input_ids, question_attention_mask)
        paragraph_embedding = self.two_tower_model(paragraph_input_ids, paragraph_attention_mask)
        cosine_sim = self.cosine_similarity(question_embedding, paragraph_embedding)
        return cosine_sim


two_tower_model = TwoTowerModel(bert_model)
if torch.cuda.device_count() > 1:
    two_tower_model = nn.DataParallel(two_tower_model)
two_tower_model = two_tower_model.to(device)

model = TwoTowerSimilarityModel(two_tower_model)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

question_chunk_dataset = QuestionChunkDataset(dataset, labels, danish_bert, max_seq_len=512, device=device)


# Define the loss function and optimizer
loss_function = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)


# Split your data into training, validation, and test sets
train_size = int(0.8 * len(question_chunk_dataset))
val_size = int(0.1 * len(question_chunk_dataset))
test_size = len(question_chunk_dataset) - train_size - val_size
train_data, val_data, test_data = random_split(question_chunk_dataset, [train_size, val_size, test_size])

# Create DataLoaders for the three sets
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False,  drop_last=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False,  drop_last=True)

def load_checkpoint(model, optimizer, checkpoints_dir, epoch):
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
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





start_epoch = 1  # Set the epoch number from which you want to continue training (1-indexed)

if start_epoch > 1:
    train_losses, test_losses = load_checkpoint(model, optimizer, checkpoints_dir, start_epoch)
else:
    train_losses, test_losses = [], []
#Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

# Fine-tune the model
num_epochs = 10
print(f"Fine-tuning the model for {num_epochs} epochs...")

best_test_loss = float('inf')  # Initialize the best validation loss as infinity
training_iterations = 0
save_checkpoint_every = 5000

for epoch in range(start_epoch - 1, num_epochs):  # Subtract 1 to make it zero-indexed

    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        question_input_ids = batch['question_input_ids'].to(device)
        question_attention_mask = batch['question_attention_mask'].to(device)
        paragraph_input_ids = batch['paragraph_input_ids'].to(device)
        paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        similarities = model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
        loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Added Gradient clipping
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_description(f"Training (loss: {loss.item():.4f})")

        training_iterations += 1
        if training_iterations % save_checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_iteration_{training_iterations}.pth')
            torch.save({
                'epoch': epoch,
                'iteration': training_iterations,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_losses,
                'test_loss_history': test_losses,
            }, checkpoint_path)

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            paragraph_input_ids = batch['paragraph_input_ids'].to(device)
            paragraph_attention_mask = batch['paragraph_attention_mask'].to(device)
            labels = batch['label'].to(device)

            similarities = model(question_input_ids, question_attention_mask, paragraph_input_ids, paragraph_attention_mask)
            loss = loss_function((similarities + 1) / 2, labels)  # Scale similarities to [0, 1] for BCELoss

            test_loss += loss.item()
            progress_bar.set_description(f"Testing (loss: {loss.item():.4f})")

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}")
    
    # Call the scheduler.step() function with the test_loss
    scheduler.step(test_loss)

    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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
        torch.save(model.state_dict(), os.path.join(save_directory, 'model.pt'))
        try:
            tokenizer.save_pretrained(save_directory)
        except:
            pass

        
