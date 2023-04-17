import os
import pickle
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.two_towers_fine_tune_multiplenegatives import *

# Define the directories to be scanned
checkpoints_dirs = [
    'models/two_tower_checkpoints_multiplenegatives_v6',
    'models/two_tower_checkpoints_multiplenegatives_v7',
    'models/two_tower_checkpoints_multiplenegatives_v8',
]

# Function to load losses from pkl files
def load_losses(folder, pattern):
    losses = []
    files = os.listdir(folder)
    for file in sorted(files, key=lambda x: int(x.split('_')[3])):
        if file.startswith(pattern) and file.endswith('.pkl'):
            with open(os.path.join(folder, file), 'rb') as f:
                losses.append(pickle.load(f))
    return losses

# Plot train and validation losses for each folder
plt.figure(figsize=(10, 5))
for i, checkpoints_dir in enumerate(checkpoints_dirs, start=1):
    train_losses = load_losses(checkpoints_dir, 'average_loss')
    x_labels = list(range(1, len(train_losses) + 1))
    
    plt.plot(x_labels, train_losses, label=f'Train loss {i}', marker='o')

plt.xlabel('Ranked Steps')
plt.ylabel('Loss')
plt.title('Evolution of Train Losses')
plt.legend()
plt.show()

# Print hyperparameters
num_epochs = len(train_losses)

print(f"Model architecture: {two_tower_model}") 
