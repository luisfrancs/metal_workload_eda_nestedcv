## VERSION 29 OCT 2025
## AUTHOR Luis Sigcha

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_histogram(df, y, feature, bins=30):
    """
    Plot histogram of one feature separated by class.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataframe
    y : numpy.ndarray
        Label vector (0/1 or two classes)
    feature : str
        Column name to plot
    bins : int
        Number of bins
    """

    classes = np.unique(y)

    plt.figure()

    for c in classes:
        plt.hist(
            df.loc[y == c, feature],
            bins=bins,
            alpha=0.6,
            label=f"Class {c}"
        )

    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {feature}")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_multiple_histograms(df, y, features, bins=30):
    """
    Plot histograms for multiple features separated by class.
    """

    classes = np.unique(y)

    for feature in features:

        plt.figure()

        for c in classes:
            plt.hist(
                df.loc[y == c, feature],
                bins=bins,
                alpha=0.6,
                label=f"Class {c}"
            )

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {feature}")
        plt.legend()
        plt.grid(True)

        plt.show()


def plot_all_subject_sequences(id_vector,y, pred_vector):
    """
    Plots the full prediction sequence for each subject in orange.
    """
    unique_ids = np.unique(id_vector) #get  unique subject
    
    for user_id in unique_ids:
        # Find all indices for this specific ID
        indices = np.where(id_vector == user_id)[0]
        # Dynamically find start and end
        start_idx = indices[0]
        end_idx = indices[-1]     
        # Extract the segment
        subject_preds = pred_vector[start_idx:end_idx+1]
        # Create the plot
        plt.figure(figsize=(10, 2))
        plt.plot(subject_preds, color='#1f77b4', linewidth=2)
        # Create the plot 2 label
        y_subject=y[start_idx:end_idx+1]
        plt.plot(y_subject, color='#ff7f0e', linewidth=2)        
        # Formatting
        plt.title(f"Full Prediction Sequence - Subject ID: {user_id}")
        plt.xlabel("Time (Samples since Subject Start)")
        plt.ylabel("Class (0/1)")
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ["Baseline (0)", "Task (1)"]) # Labelling the binary classes
        plt.grid(axis='both', linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        plt.show()