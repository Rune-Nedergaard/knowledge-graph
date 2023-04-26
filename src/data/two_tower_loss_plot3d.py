import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

models_path = "models"
subfolder_names = [f"two_tower_checkpoints_multiplenegatives_v{i}" for i in range(10, 20)]

# Initialize dictionaries to store loss data
train_losses = {i: [] for i in range(1, 5)}
val_losses = {i: [] for i in range(1, 5)}

# Loop through the subfolders
for subfolder in subfolder_names:
    for epoch in range(1, 5):
        train_loss_file = os.path.join(models_path, subfolder, f"train_losses_epoch_{epoch}.pkl")
        val_loss_file = os.path.join(models_path, subfolder, f"val_losses_epoch_{epoch}.pkl")

        # Check if files exist
        if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
            # Load the losses from the pickle files
            with open(train_loss_file, "rb") as f:
                train_loss_data = pickle.load(f)
            with open(val_loss_file, "rb") as f:
                val_loss_data = pickle.load(f)

            # Append the losses to the corresponding dictionaries
            train_losses[epoch].append(train_loss_data)
            val_losses[epoch].append(val_loss_data)

# Set seaborn style
sns.set_style("whitegrid")

# Create a 3D plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.get_cmap('tab10', len(subfolder_names))

# Define a range of markers
markers = ['o', 'v', 's', 'p', '*', 'h', 'D', 'd', 'P', 'X']

# Set perturbation range
perturbation_range = 0.005

for epoch in range(1, 5):
    for i, (train_loss, val_loss) in enumerate(zip(train_losses[epoch], val_losses[epoch])):
        model_version = subfolder_names[i][-3:]
        color = colors(i % len(subfolder_names))
        
        perturbation = np.random.uniform(-perturbation_range, perturbation_range)
        y_values_train = [epoch + perturbation] * len(train_loss)
        y_values_val = [epoch + perturbation] * len(val_loss)
        
        ax.plot(range(1, len(train_loss) + 1), y_values_train, train_loss, color=color, linestyle='-', label=f"Train Loss {model_version}" if epoch == 1 else None, markersize=3, marker=markers[i], alpha=0.5)
        ax.plot(range(1, len(val_loss) + 1), y_values_val, val_loss, color=color, linestyle='--', label=f"Val Loss {model_version}" if epoch == 1 else None, markersize=3, marker=markers[i], alpha=0.5)

# Customize the plot
ax.set_xlabel("Epoch")
ax.set_xticks(range(1, 5))
ax.set_ylabel("")
ax.set_ylim(2.99,3.01)
#remove yticks
ax.set_yticks([])
ax.set_zlabel("Loss")
ax.set_title("Training and Validation Losses")


# Create a custom legend with separate entries for training and validation loss
handles, labels = ax.get_legend_handles_labels()
legend_elements = []
for i in range(len(subfolder_names)):
    model_version = subfolder_names[i][-3:]
    color = colors(i % len(subfolder_names))
    legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='-', label=f"Train Loss {model_version}"))
    legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', label=f"Val Loss {model_version}"))

# Since 3D plots do not have built-in legend support, we need to manually create a legend using a 2D plot
legend_fig, legend_ax = plt.subplots(figsize=(6, 3))
legend_ax.axis('off')
legend_ax.legend(handles=legend_elements, ncol=2, loc='center')
legend_fig.suptitle("Legend")
legend_fig.show()

# Show the plot
plt.show()
