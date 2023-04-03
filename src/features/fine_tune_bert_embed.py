from danlp.models import load_bert_base_model
from torch.utils.data import Dataset
import pickle
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.init as init
#from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Create a directory to save checkpoints
checkpoints_dir = 'models/checkpoints'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
danish_bert = load_bert_base_model()


tokenizer = danish_bert.tokenizer
bert_model = danish_bert.model

dataset = pickle.load(open('data/fine_tune_dataset.pkl', 'rb'))
labels = pickle.load(open('data/fine_tune_labels.pkl', 'rb'))

def embed_text(self, text_a, text_b=None):
    """
    Calculate the embeddings for each token in a sentence and the embedding for the sentence based on a BERT language model.
    The embedding for a token is chosen to be the concatenated last four layers, and the sentence embeddings to be the mean of the second to last layer of all tokens in the sentence
    The BERT tokenizer splits in subword for UNK word. The tokenized sentence is therefore returned as well. The embeddings for the special tokens are not returned.
   
    :param str text_a: raw text (first sentence)
    :param str text_b: raw text (second sentence, optional)
    :return: three lists: token_embeddings (dim: tokens x 3072), sentence_embedding (1x738), tokenized_text
    :rtype: list, list, list
    """

    # If there is a second sentence, concatenate them with the separator
    if text_b:
        marked_text = "[CLS] " + text_a + " [SEP] " + text_b + " [SEP]"
    else:
        marked_text = "[CLS] " + text_a + " [SEP]"

    # Tokenize sentence with the BERT tokenizer
    tokenized_text = self.tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces
    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the tokens as belonging to sentence "1"
    if text_b:
        sep_index = tokenized_text.index("[SEP]")
        segments_ids = [0] * (sep_index + 1) + [1] * (len(tokenized_text) - sep_index - 1)
    else:
        segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = self.model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1. to tokens x layers x embedding
    token_embeddings = token_embeddings.permute(1,0,2)

    # choose to concatenate last four layers, dim 4x 768 = 3072
    token_vecs_cat = [torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) for token in token_embeddings]
    # drop the CLS and the SEP tokens and embedding
    token_vecs_cat = token_vecs_cat[1:-1]
    tokenized_text = tokenized_text[1:-1]

    # chose to summarize the last four layers
    # token_vecs_sum=[torch.sum(token[-4:], dim=0) for token in token_embeddings]

    # sentence embedding
    # Calculate the average of all token vectors for the second last layers
    sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

    return token_vecs_cat, sentence_embedding, tokenized_text


class QuestionChunkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.labels)

    # Correct the indentation here
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

        # He initialization
        init.kaiming_normal_(self.linear.weight)

        # Xavier initialization
        # init.xavier_normal_(self.linear.weight)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the [CLS] token representation
        logits = self.linear(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities.squeeze()


model = SemanticSimilarityModel(bert_model)
model = model.to(device)

# Define the loss function and optimizer
loss_function = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)


# Split your data into training, validation, and test sets
train_size = int(0.8 * len(question_chunk_dataset))
val_size = int(0.1 * len(question_chunk_dataset))
test_size = len(question_chunk_dataset) - train_size - val_size
train_data, val_data, test_data = random_split(question_chunk_dataset, [train_size, val_size, test_size])

# Create DataLoaders for the three sets
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

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





start_epoch = 3  # Set the epoch number from which you want to continue training (1-indexed)
train_losses, test_losses = load_checkpoint(model, optimizer, checkpoints_dir, start_epoch)

#Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
print(f"Using learning rate scheduler: {scheduler}")


if train_losses is None:
    train_losses = []
    test_losses = []
    start_epoch = 1
else:
    print(f"Continuing training from epoch {start_epoch}")

# Fine-tune the model
num_epochs = 6
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

    # Save the model if the test loss is lower than the best test loss
    if test_loss < best_test_loss:
        print(f"Improved test loss from {best_test_loss:.4f} to {test_loss:.4f}. Saving the model.")
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'models/fine_tuned_model.pth')


#save test loss and train loss
import pickle
with open('data/train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
with open('data/test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)


# Plot training and test losses
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Test Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
