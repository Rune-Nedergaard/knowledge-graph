import pickle
import matplotlib.pyplot as plt

# Load train_losses and test_losses
with open('data/train_losses.pkl', 'rb') as f:
    train_losses = pickle.load(f)
with open('data/test_losses.pkl', 'rb') as f:
    test_losses = pickle.load(f)

# Load old_ce_train_losses and old_ce_test_losses
with open('data/old_ce_train_losses.pkl', 'rb') as f:
    old_ce_train_losses = pickle.load(f)
with open('data/old_ce_test_losses.pkl', 'rb') as f:
    old_ce_test_losses = pickle.load(f)

# Set plot style and size
plt.style.use('seaborn')
plt.figure(figsize=(12, 8))

# Plot the losses
plt.plot(train_losses, label='Batch size: 30, Filtered Data, Training Loss', linestyle='-', linewidth=2)
plt.plot(test_losses, label='Batch size: 30, Filtered Data, Test Loss', linestyle='--', linewidth=2)
plt.plot(old_ce_train_losses, label='Batch size: 90, Unfiltered Data, Training Loss', linestyle='-', linewidth=2, color='C2')
plt.plot(old_ce_test_losses, label='Batch size: 90, Unfiltered Data, Test Loss', linestyle='--', linewidth=2, color='C2')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Cross-Encoder Training and Test Losses')
plt.savefig('losses_plot.png')
plt.show()
