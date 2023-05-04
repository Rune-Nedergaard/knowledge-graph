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

# Create a directory to save checkpoints
checkpoints_dir = 'models/checkpoints_fine_tune_bert_embed_hpc'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Set device and check for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
    print(f"Using device: {device}")

danish_bert = load_bert_base_model()

tokenizer = danish_bert.tokenizer
bert_model = danish_bert.model

dataset = pickle.load(open('data/fine_tune_dataset.pkl', 'rb'))
labels = pickle.load(open('data/fine_tune_labels.pkl', 'rb'))


class QuestionChunkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        question, paragraph = self.texts[idx]  # Use self.texts instead of self.question_chunk_pairs

        # Tokenize question and paragraph
        question_tokens = self.tokenizer.tokenize(question)
        paragraph_tokens = self.tokenizer.tokenize(paragraph)

        # Insert special tokens
        tokens = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + paragraph_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    
        # Truncate or pad the token ids
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))

        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Labels (1 for relevant, 0 for not relevant)
question_chunk_dataset = QuestionChunkDataset(dataset, labels, tokenizer)

# print(f"Length of dataset: {len(question_chunk_dataset)}")

# # Check the length of the dataset
# print(f"Length of dataset: {len(question_chunk_dataset)}")

# # Check a few samples from the dataset
# for idx in range(5):  # You can change the range to check more or fewer samples
#     sample = question_chunk_dataset[idx]
#     print(f"Sample {idx+1}:")
    
#     input_ids = sample['input_ids']
#     attention_mask = sample['attention_mask']
#     label = sample['label']

#     # Print input_ids and decoded text
#     print("  input_ids:", input_ids)
    
#     # Decode the combined question and paragraph
#     decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
#     print("  Decoded text:", decoded_text)
    
#     # Print attention_mask
#     print("  attention_mask:", attention_mask)
    
#     # Print label
#     print("  label:", label)


class SemanticSimilarityModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # Xavier initialization
        init.xavier_normal_(self.linear.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        logits = self.linear(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities.squeeze() # Reshape from (batch_size, 1) to (batch_size,)


# Use DataParallel if multiple GPUs are available
model = SemanticSimilarityModel(bert_model)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Define the loss function and optimizer
loss_function = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.005)


# Split your data into training, validation, and test sets
train_size = int(0.8 * len(question_chunk_dataset))
val_size = int(0.1 * len(question_chunk_dataset))
test_size = len(question_chunk_dataset) - train_size - val_size
train_data, val_data, test_data = random_split(question_chunk_dataset, [train_size, val_size, test_size])

# Create DataLoaders for the three sets
train_loader = DataLoader(train_data, batch_size=80, shuffle=True)
val_loader = DataLoader(val_data, batch_size=80, shuffle=False)
test_loader = DataLoader(test_data, batch_size=80, shuffle=False)

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
# Moved these indside the load function
#train_losses = []  # Initialize list to store training losses for each epoch
#test_losses = []  # Initialize list to store test losses for each epoch

#for epoch in range(num_epochs):
for epoch in range(start_epoch - 1, num_epochs):  # Subtract 1 to make it zero-indexed

    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        probabilities = model(input_ids, attention_mask)
        loss = loss_function(probabilities, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #added Gradient clipping
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_description(f"Training (loss: {loss.item():.4f})")

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
        
    # Test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            probabilities = model(input_ids, attention_mask)
            loss = loss_function(probabilities, labels.float())

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
    with open('data/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('data/test_losses.pkl', 'wb') as f:
        pickle.dump(test_losses, f)


    # Save the model if the test loss is lower than the best test loss
    if test_loss < best_test_loss:
        print(f"Improved test loss from {best_test_loss:.4f} to {test_loss:.4f}. Saving the model.")
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'models/reranker_model.pth')




# Plot training and test losses
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
