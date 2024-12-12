import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_next_epoch_number(base_path):
    """Get the next available epoch number from existing files."""
    pattern = f"{base_path}-epoch*.png"
    existing_files = glob.glob(pattern)
    if not existing_files:
        return 1

    numbers = [int(f.split("epoch")[-1].split(".")[0]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1


def plot_attention_weights(attn_weights_dict, save_dir="attention_plots"):
    """
    Plot and save heatmaps for attention weights from transformer layers.
    Only plots the first 4 samples from the last batch.

    Args:
        attn_weights_dict : a dictionary containing attention weights for each layer
        save_dir (str): Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)

    for attn_name, attn_weights in attn_weights_dict.items():
        # Get the last batch's attention weights
        weights = attn_weights.detach().cpu()

        # Plot first 4 samples
        for sample_idx in range(min(4, weights.shape[0])):
            # Create figure and axis
            plt.figure(figsize=(20, 16))

            # Plot heatmap using seaborn
            sns.heatmap(
                weights[sample_idx].numpy(),
                cmap="viridis",
                cbar_kws={"label": "Attention Weight"},
            )

            # Set title and labels
            plt.title(f"{attn_name} (Sample {sample_idx+1})")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")

            # Get base path and next epoch number
            base_path = os.path.join(save_dir, f"sample{sample_idx+1}_{attn_name}")
            epoch_num = get_next_epoch_number(base_path)

            # Save plot with epoch suffix
            save_path = f"{base_path}-epoch{epoch_num}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
