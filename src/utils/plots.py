import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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