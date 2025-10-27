import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils.utils import make_dir


def plot_avg_confidence(df, group_col, group_labels, title):
    """
    Plots average placement and seen-when confidence ratings for a binary grouping.
    
    Parameters:
        df: pandas DataFrame
        group_col: str, column name to group by (should be 0/1 or boolean)
        group_labels: list of str, labels for 0 and 1
        title: str, plot title
    """
    means_placement = df.groupby(group_col)['placement_cr'].mean() / 100
    means_seen_when = df.groupby(group_col)['seen_when_cr'].mean() / 100

    x = np.arange(len(group_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, means_placement, width, label='Placement Confidence Rating', color='b')
    bars2 = ax.bar(x + width/2, means_seen_when, width, label='Seen When Confidence Rating', color='g')

    ax.set_ylabel('Average Confidence Rating')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

def plot_bar(labels, values, errors=None, title="", ylabel="Spearman ρ", path=None, ylim=(-1, 1), colors=None, scale=1e1):
    plt.figure(figsize=(6, 4))
    
    # Handle NaNs and scale values
    values = np.nan_to_num(values) * scale
    if errors is not None:
        errors = np.nan_to_num(errors) * scale

    # Plot
    plt.bar(labels, values, yerr=errors, capsize=5 if errors is not None else 0, color=colors)
    plt.ylim(*ylim)
    
    # Label the axis with scale note
    plt.ylabel(f"{ylabel} (×10⁻¹)")
    plt.title(title)
    
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
    plt.close()

def plot_split(sub, spat_low, spat_high, path):
    plt.figure(figsize=(8, 4))
    # Spatial
    plt.bar(['Low', 'High'], [spat_low, spat_high], color=['salmon', 'red'])
    plt.ylim(-1, 1)
    plt.title(f"Spatial RDM ↔ RSA ({sub})")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    make_dir(path)
    plt.savefig(os.path.join(path, f"{sub}_rsa_median_split.png"))
    plt.close()